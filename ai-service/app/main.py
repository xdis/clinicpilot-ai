"""FastAPI entry point for the ClinicPilot AI service.

    uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from langgraph.checkpoint.sqlite import SqliteSaver

from .agent import ClinicAgent
from .backend_client import BackendClient, ToolError
from .config import Settings, get_settings
from .knowledge import KnowledgeBase
from .llm import build_llm
from .schemas import ChatRequest, ChatResponse, MedicalSummary, SummarizeRequest
from .tools import McpTools, RestTools, ToolGateway

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("clinicpilot.ai")


def build_agent(settings: Settings) -> ClinicAgent:
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
    return ClinicAgent(
        llm=llm,
        tools_factory=tools_factory,
        kb_loader=kb_loader,
        checkpointer=SqliteSaver(conn),
        clinic_timezone=settings.clinic_timezone,
        clinic_today=settings.clinic_today,
        history_turns=settings.history_turns,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.agent = build_agent(settings)
    app.state.lock = threading.Lock()  # one turn at a time: simple and safe for a single-patient demo
    log.info("AI service ready: llm=%s tools=%s backend=%s", app.state.agent.llm.name, settings.tool_transport, settings.backend_url)
    yield


app = FastAPI(title="ClinicPilot AI Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    s: Settings = request.app.state.settings
    return {"status": "ok", "llm": request.app.state.agent.llm.name, "tool_transport": s.tool_transport}


@app.post("/v1/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    agent: ClinicAgent = request.app.state.agent
    with request.app.state.lock:
        try:
            result = agent.chat(body.session_id, body.patient_id, body.message)
        except ToolError as exc:
            log.error("tool failure: %s %s", exc.code, exc.message)
            raise HTTPException(status_code=502, detail={"code": exc.code, "message": exc.message}) from exc
    log.info("chat session=%s intent=%s tools=%s latency=%sms", body.session_id, result["intent"],
             [c["name"] for c in result["tool_calls"]], result["latency_ms"])
    return ChatResponse(**result)


@app.post("/v1/sessions/{session_id}/reset")
def reset(session_id: str, request: Request, patient_id: str = "P001") -> dict:
    with request.app.state.lock:
        request.app.state.agent.reset(session_id, patient_id)
    return {"status": "reset", "session_id": session_id}


@app.post("/v1/summarize")
def summarize(body: SummarizeRequest, request: Request) -> dict:
    agent: ClinicAgent = request.app.state.agent
    started = time.perf_counter()
    agent.llm.usage = {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}
    try:
        summary: MedicalSummary = agent.llm.summarize(body.transcript)
    except Exception as exc:  # noqa: BLE001 - one retry, then a clean error
        log.warning("summarize retry after: %s", exc)
        try:
            summary = agent.llm.summarize(body.transcript)
        except Exception as exc2:  # noqa: BLE001
            raise HTTPException(status_code=502, detail={"code": "SUMMARY_FAILED", "message": str(exc2)}) from exc2
    return {"summary": summary.model_dump(), "usage": dict(agent.llm.usage), "latency_ms": int((time.perf_counter() - started) * 1000)}
