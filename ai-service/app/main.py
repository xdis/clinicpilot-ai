"""FastAPI entry point for the ClinicPilot AI service.

    uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from .agent import ClinicAgent
from .backend_client import BackendClient, ToolError, current_request_id
from .config import Settings, get_settings
from .knowledge import KnowledgeBase
from .llm import build_llm
from .memory import PatientMemory, ShortTermMemory
from .schemas import ChatRequest, ChatResponse, FeedbackRequest, MedicalSummary, SummarizeRequest
from .telemetry import Observability
from .transcribe import AUDIO_TYPES, MAX_AUDIO_BYTES, Transcriber, TranscriberUnavailable
from .tools import McpTools, RestTools, ToolGateway

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("clinicpilot.ai")


def build_agent(settings: Settings, obs: Observability | None = None) -> ClinicAgent:
    llm = build_llm(settings)
    backend = BackendClient(settings.backend_url)

    def tools_factory() -> ToolGateway:
        if settings.tool_transport == "mcp":
            return McpTools(settings.backend_url)
        return RestTools(backend)

    def kb_loader(tools: ToolGateway) -> KnowledgeBase:
        return KnowledgeBase(tools.call("get_catalog"))

    Path(settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
    memory = None
    if settings.memory_enabled:
        Path(settings.memory_db).parent.mkdir(parents=True, exist_ok=True)
        store = SqliteStore(sqlite3.connect(settings.memory_db, check_same_thread=False, isolation_level=None))
        store.setup()
        memory = PatientMemory(store, tenant=settings.memory_tenant, retention_days=settings.memory_retention_days)
    return ClinicAgent(
        llm=llm,
        tools_factory=tools_factory,
        kb_loader=kb_loader,
        checkpointer=SqliteSaver(conn),
        clinic_timezone=settings.clinic_timezone,
        clinic_today=settings.clinic_today,
        history_turns=settings.history_turns,
        short_term=ShortTermMemory(settings.short_term_max_tokens, max_messages=2 * settings.history_turns),
        memory=memory,
        memory_llm=build_llm(settings),  # own usage counters: background writes don't skew per-turn stats
        obs=obs or Observability(),
        expose_grounding=settings.eval_expose_grounding,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.agent = build_agent(settings, OBS)
    app.state.transcriber = Transcriber(settings.whisper_model) if Transcriber.available() else None
    if app.state.transcriber is not None and settings.whisper_preload:
        threading.Thread(target=app.state.transcriber.preload, daemon=True).start()
    app.state.lock = threading.Lock()  # one turn at a time: simple and safe for a single-patient demo
    log.info("AI service ready: llm=%s tools=%s backend=%s langfuse=%s otel=%s", app.state.agent.llm.name,
             settings.tool_transport, settings.backend_url, OBS.langfuse is not None, OBS.otel)
    yield
    OBS.shutdown()  # flush pending traces and scores


# Tracing is set up before the app and the LLM clients exist: the OpenTelemetry provider must be registered first,
# and FastAPI is instrumented at creation time. Both are no-ops unless configured (see app/telemetry.py).
OBS = Observability(get_settings())
app = FastAPI(title="ClinicPilot AI Service", version="1.0.0", lifespan=lifespan)
OBS.instrument_app(app)


@app.get("/health")
def health(request: Request) -> dict:
    s: Settings = request.app.state.settings
    t = request.app.state.transcriber
    return {"status": "ok", "llm": request.app.state.agent.llm.name, "tool_transport": s.tool_transport,
            "memory": request.app.state.agent.memory is not None,
            "tracing": {"langfuse": OBS.langfuse is not None, "otel": OBS.otel},
            "speech_to_text": None if t is None else {"model": t.model_size, "loaded": t._model is not None}}


@app.post("/v1/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    agent: ClinicAgent = request.app.state.agent
    rid = request.headers.get("x-request-id", "")
    current_request_id.set(rid if re.fullmatch(r"[A-Za-z0-9-]{1,64}", rid) else None)
    with request.app.state.lock:
        try:
            result = agent.chat(body.session_id, body.patient_id, body.message)
        except ToolError as exc:
            log.error("tool failure: %s %s", exc.code, exc.message)
            raise HTTPException(status_code=502, detail={"code": exc.code, "message": exc.message}) from exc
    log.info("chat request_id=%s session=%s intent=%s tools=%s latency=%sms", rid or "-", body.session_id,
             result["intent"], [c["name"] for c in result["tool_calls"]], result["latency_ms"])
    return ChatResponse(**result)


@app.post("/v1/sessions/{session_id}/reset")
def reset(session_id: str, request: Request, patient_id: str = "P001") -> dict:
    _check_ids(session_id, patient_id)
    with request.app.state.lock:
        request.app.state.agent.reset(session_id, patient_id)
    return {"status": "reset", "session_id": session_id}


@app.post("/v1/sessions/{session_id}/end")
def end_session(session_id: str, request: Request, patient_id: str = "P001") -> dict:
    """The patient left or started a new conversation: write one long-term episode in the background."""
    _check_ids(session_id, patient_id)
    with request.app.state.lock:
        scheduled = request.app.state.agent.end_session(session_id, patient_id)
    return {"status": "ended", "session_id": session_id, "episode_scheduled": scheduled}


@app.get("/v1/patients/{patient_id}/memory")
def get_memory(patient_id: str, request: Request) -> dict:
    """What the assistant remembers about a patient (for the patient, staff and audits)."""
    _check_ids("x", patient_id)
    agent: ClinicAgent = request.app.state.agent
    if agent.memory is None:
        return {"enabled": False}
    return {"enabled": True, **agent.memory.snapshot(patient_id)}


@app.delete("/v1/patients/{patient_id}/memory")
def forget(patient_id: str, request: Request) -> dict:
    """Right to erasure: deletes long-term memory and every checkpointed session of the patient."""
    _check_ids("x", patient_id)
    agent: ClinicAgent = request.app.state.agent
    with request.app.state.lock:
        deleted = agent.forget_patient(patient_id)
    return {"status": "forgotten", "patient_id": patient_id, "deleted": deleted}


def _check_ids(session_id: str, patient_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id) or not re.fullmatch(r"P\d{3,}", patient_id):
        raise HTTPException(status_code=422, detail={"code": "INVALID_ID", "message": "invalid session or patient id"})


@app.post("/v1/feedback")
def feedback(body: FeedbackRequest) -> dict:
    """User feedback (e.g. a UI thumbs up/down, not wired yet), stored as a Langfuse score on the turn's trace (online evaluation signal)."""
    OBS.score(body.trace_id, "user_feedback", 1 if body.value else 0, data_type="BOOLEAN", comment=body.comment)
    return {"status": "recorded" if OBS.langfuse is not None else "tracing_disabled"}


def _summarize(request: Request, transcript: str) -> tuple[MedicalSummary, dict]:
    agent: ClinicAgent = request.app.state.agent
    with request.app.state.lock:  # shares the LLM usage counters with chat turns
        agent.llm.usage = {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}
        try:
            summary: MedicalSummary = agent.llm.summarize(transcript)
        except Exception as exc:  # noqa: BLE001 - one retry, then a clean error
            log.warning("summarize retry after: %s", exc)
            try:
                summary = agent.llm.summarize(transcript)
            except Exception as exc2:  # noqa: BLE001
                raise HTTPException(status_code=502, detail={"code": "SUMMARY_FAILED", "message": str(exc2)}) from exc2
        return summary, dict(agent.llm.usage)


@app.post("/v1/summarize")
def summarize(body: SummarizeRequest, request: Request) -> dict:
    started = time.perf_counter()
    with OBS.job("clinicpilot.summarize", input={"chars": len(body.transcript)}):
        summary, usage = _summarize(request, body.transcript)
    return {"summary": summary.model_dump(), "usage": usage, "latency_ms": int((time.perf_counter() - started) * 1000)}


@app.post("/v1/summarize/audio")
async def summarize_audio(request: Request, file: UploadFile = File(...)) -> dict:
    """Bonus: consultation recording -> local Whisper transcript -> structured medical summary."""
    transcriber: Transcriber | None = request.app.state.transcriber
    if transcriber is None:
        raise HTTPException(status_code=501, detail={"code": "TRANSCRIPTION_UNAVAILABLE",
                                                     "message": "Speech-to-text is not installed (pip install faster-whisper)."})
    if int(request.headers.get("content-length") or 0) > MAX_AUDIO_BYTES + 64 * 1024:  # before reading the body
        raise HTTPException(status_code=413, detail={"code": "AUDIO_TOO_LARGE", "message": "Recording is larger than 15 MB."})
    if not (file.content_type or "").startswith(AUDIO_TYPES):
        raise HTTPException(status_code=415, detail={"code": "UNSUPPORTED_MEDIA", "message": f"Expected audio, got {file.content_type}."})
    audio = await file.read(MAX_AUDIO_BYTES + 1)
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail={"code": "AUDIO_TOO_LARGE", "message": "Recording is larger than 15 MB."})
    if not audio:
        raise HTTPException(status_code=422, detail={"code": "EMPTY_AUDIO", "message": "The recording is empty."})
    started = time.perf_counter()
    with OBS.job("clinicpilot.summarize_audio", input={"bytes": len(audio), "content_type": file.content_type}):
        with OBS.span("transcribe", metadata={"model": f"faster-whisper-{transcriber.model_size}"}) as span:
            try:  # CPU-bound: run off the event loop; the transcriber has its own lock
                stt = await run_in_threadpool(transcriber.transcribe, audio, file.filename or "audio")
            except TranscriberUnavailable as exc:  # model could not be loaded / downloaded: our problem, not the file's
                log.error("speech-to-text unavailable: %s", exc)
                raise HTTPException(status_code=503, detail={"code": "TRANSCRIPTION_UNAVAILABLE",
                                                             "message": "Speech-to-text is temporarily unavailable."}) from exc
            except Exception as exc:  # noqa: BLE001 - unreadable audio, codec errors
                log.warning("transcription failed: %s", exc)
                raise HTTPException(status_code=422, detail={"code": "TRANSCRIPTION_FAILED",
                                                             "message": "The recording could not be decoded."}) from exc
            if span is not None:
                span.update(output={"language": stt["language"], "audio_seconds": stt["audio_seconds"]})
        if not stt["transcript"].strip():
            raise HTTPException(status_code=422, detail={"code": "NO_SPEECH", "message": "No speech was detected in the recording."})
        summary, usage = await run_in_threadpool(_summarize, request, stt["transcript"])
    return {**stt, "summary": summary.model_dump(), "usage": usage,
            "latency_ms": int((time.perf_counter() - started) * 1000)}
