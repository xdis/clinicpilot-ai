"""Observability: OpenTelemetry for service traces, Langfuse for LLM / agent traces, one trace id.

    browser -> NestJS (OTel, undici) --traceparent--> FastAPI (OTel) -> LangGraph turn (Langfuse "agent")
                                                                        -> nodes, LLM generations (CallbackHandler)
                                                                        -> background memory writes (linked spans)

- OpenTelemetry (`OTEL_EXPORTER_OTLP_ENDPOINT` set): a global TracerProvider exporting OTLP/HTTP (to the OTel
  Collector -> Jaeger), FastAPI and httpx auto-instrumented. Standard OTEL_* env vars apply (service name, headers).
- Langfuse (`LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY`): its own TracerProvider, so prompts and patient text go
  to Langfuse only and never to the general-purpose trace backend. Spans still share the request's trace id because
  OpenTelemetry context is process-wide.
- A mask function runs on every input/output before it leaves the process (emails, phone numbers, dates of birth).
- Everything is optional and fail-safe: without configuration, or if a backend is down, the agent works unchanged.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import threading
from contextvars import ContextVar
from typing import Any, Iterator

log = logging.getLogger(__name__)

# LangChain callbacks for LLM calls made outside the LangGraph run (background memory writes run in a thread
# where the graph's callback context does not exist). llm.py reads this.
llm_callbacks: ContextVar[list | None] = ContextVar("llm_callbacks", default=None)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<![\w-])\+?\d[\d\s().-]{7,}\d(?![\w-])")
_DOB = re.compile(r"\b(born|dob|date of birth)\b[\s:]*\d{4}-\d{2}-\d{2}", re.I)


# Patient-record fields carried in graph state (the profile) are redacted by key.
_PII_KEYS = {"fullName", "firstName", "lastName", "dob", "dateOfBirth", "email", "phone", "emergencyContact", "address"}


def mask_pii(*, data: Any, **_: Any) -> Any:
    """Langfuse mask hook: redacts direct identifiers in any nested structure before export.

    By key: the profile's name, date of birth, e-mail, phone, emergency contact. By pattern, in free text: e-mails,
    phone numbers (clinic numbers too: a simple, safe rule), "born <date>". Patient ids (P001) are pseudonymous and
    kept so traces can be found. Not masked: the patient's first name inside prompt text ("talking to John") and
    allergies, which the answer needs; that is why Langfuse is self-hosted and gets its own trace pipeline.
    """
    if isinstance(data, dict):
        return {k: ("[redacted]" if k in _PII_KEYS and v else mask_pii(data=v)) for k, v in data.items()}
    if isinstance(data, str):
        data = _EMAIL.sub("[email]", data)
        data = _DOB.sub(lambda m: f"{m.group(1)} [date]", data)
        # 9+ digits: phone numbers, not ISO dates (8 digits) or times
        return _PHONE.sub(lambda m: "[phone]" if sum(c.isdigit() for c in m.group(0)) >= 9 else m.group(0), data)
    if isinstance(data, (list, tuple)):
        return [mask_pii(data=v) for v in data]
    return data


def pseudonym(patient_id: str, salt: str = "") -> str:
    """Stable, non-reversible user id for traces (Langfuse user_id must be ASCII, <= 200 chars)."""
    return "u_" + hashlib.sha256(f"{salt}:{patient_id}".encode()).hexdigest()[:16]


class Observability:
    """Thin facade the agent uses; every method is a no-op when the backend is not configured."""

    def __init__(self, settings: Any | None = None):
        self.langfuse = None
        self.otel = False
        self._salt = getattr(settings, "langfuse_user_salt", "") or ""
        self._lock = threading.Lock()
        self._trace_url_prefix: str | None = None  # ".../project/<id>/traces/", resolved once in the background
        self._public_key: str | None = None
        if settings is not None:
            self._setup_otel()
            self._setup_langfuse(settings)

    # setup --------------------------------------------------------------------------------
    def _setup_otel(self) -> None:
        if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTEL_SDK_DISABLED") == "true":
            return
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider(resource=Resource.create({
                "service.name": os.environ.get("OTEL_SERVICE_NAME", "clinicpilot-ai-service")}))
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # endpoint/headers from OTEL_* env
            trace.set_tracer_provider(provider)
            HTTPXClientInstrumentor().instrument()  # tool calls to the backend carry traceparent
            self.otel = True
            log.info("OpenTelemetry enabled -> %s", os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"])
        except Exception as exc:  # noqa: BLE001 - telemetry must never break the service
            log.warning("OpenTelemetry disabled: %s", exc)

    def _setup_langfuse(self, s: Any) -> None:
        if not (s.langfuse_public_key and s.langfuse_secret_key):
            return
        try:
            from langfuse import Langfuse
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider

            self.langfuse = Langfuse(
                public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host or None,
                environment=s.langfuse_environment, release=s.app_release, mask=mask_pii,
                # own provider: Langfuse spans (with prompts) are not sent to the general OTel backend
                tracer_provider=TracerProvider(resource=Resource.create({"service.name": "clinicpilot-ai-service"})),
            )
            self._public_key = s.langfuse_public_key
            threading.Thread(target=self._resolve_trace_url_prefix, daemon=True).start()
            log.info("Langfuse enabled (%s)", s.langfuse_host or "default host")
        except Exception as exc:  # noqa: BLE001
            log.warning("Langfuse disabled: %s", exc)
            self.langfuse = None

    def instrument_app(self, app: Any) -> None:
        if not self.otel:
            return
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="health")

    # per turn -------------------------------------------------------------------------------
    def handler(self, trace_context: dict | None = None) -> list:
        """LangChain callbacks to pass to graph.invoke (nodes and LLM generations become child observations)."""
        if self.langfuse is None:
            return []
        from langfuse.langchain import CallbackHandler

        # explicit public key: the handler must use this client, not whichever Langfuse client was created first
        kw = {"public_key": self._public_key, **({"trace_context": trace_context} if trace_context else {})}
        return [CallbackHandler(**kw)]

    @contextlib.contextmanager
    def turn(self, *, session_id: str, patient_id: str, message: str, tags: list[str] | None = None) -> Iterator[Any]:
        """One chat turn = one Langfuse trace (agent observation) carrying session, pseudonymous user and tags."""
        if self.langfuse is None:
            yield None
            return
        from langfuse import propagate_attributes
        from opentelemetry import trace

        from .backend_client import service_span

        server = trace.get_current_span()  # the FastAPI request span (OpenTelemetry), before Langfuse takes over
        token = service_span.set(server if server.get_span_context().is_valid else None)
        try:
            with propagate_attributes(session_id=session_id, user_id=pseudonym(patient_id, self._salt),
                                      tags=tags or ["chat"], trace_name="clinicpilot.chat_turn"):
                with self.langfuse.start_as_current_observation(name="clinicpilot.chat_turn", as_type="agent",
                                                                 input={"message": message}) as obs:
                    yield obs
        finally:
            service_span.reset(token)

    def current_trace(self) -> dict | None:
        """Trace id + url of the current turn, for the API response and for linking background work."""
        if self.langfuse is None:
            return None
        tid = self.langfuse.get_current_trace_id()
        if not tid:
            return None
        return {"trace_id": tid, "trace_url": self._trace_url(tid)}

    def _resolve_trace_url_prefix(self) -> None:
        """get_trace_url() calls the Langfuse API (with retries) for the project id: done once, off the hot path."""
        try:
            url = self.langfuse.get_trace_url(trace_id="0" * 32)  # type: ignore[union-attr]
            self._trace_url_prefix = url.rsplit("/", 1)[0] + "/" if url else None
        except Exception as exc:  # noqa: BLE001 - Langfuse down: turns simply carry no link
            log.warning("Langfuse trace url unavailable: %s", exc)

    def _trace_url(self, trace_id: str) -> str | None:
        return self._trace_url_prefix + trace_id if self._trace_url_prefix else None

    def score(self, trace_id: str | None, name: str, value: Any, data_type: str = "NUMERIC", comment: str | None = None) -> None:
        if self.langfuse is None or not trace_id:
            return
        try:
            self.langfuse.create_score(trace_id=trace_id, name=name, value=value, data_type=data_type, comment=comment)
        except Exception as exc:  # noqa: BLE001
            log.warning("Langfuse score failed: %s", exc)

    @contextlib.contextmanager
    def background(self, name: str, trace_id: str | None, *, session_id: str, patient_id: str,
                   input: Any = None) -> Iterator[None]:
        """Span for background memory work (runs in another thread). Attached to the turn that triggered it when
        trace_id is given, otherwise its own trace (e.g. the episode written when a conversation ends). Session and
        pseudonymous user are set again: thread-local context does not cross into the worker."""
        if self.langfuse is None:
            yield
            return
        from langfuse import propagate_attributes

        ctx = {"trace_context": {"trace_id": trace_id}} if trace_id else {}
        with propagate_attributes(session_id=session_id, user_id=pseudonym(patient_id, self._salt), tags=["memory"]), \
                self.langfuse.start_as_current_observation(name=name, as_type="span", input=input, **ctx):
            token = llm_callbacks.set(self.handler())
            try:
                yield
            finally:
                llm_callbacks.reset(token)

    @contextlib.contextmanager
    def job(self, name: str, *, input: Any = None, as_type: str = "chain") -> Iterator[Any]:
        """A traced unit of work outside the chat graph (e.g. the consultation summarizer); LLM calls inside nest."""
        if self.langfuse is None:
            yield None
            return
        with self.langfuse.start_as_current_observation(name=name, as_type=as_type, input=input) as obs:  # type: ignore[arg-type]
            token = llm_callbacks.set(self.handler())
            try:
                yield obs
            finally:
                llm_callbacks.reset(token)

    @contextlib.contextmanager
    def span(self, name: str, **kw: Any) -> Iterator[Any]:
        if self.langfuse is None:
            yield None
            return
        with self.langfuse.start_as_current_observation(name=name, as_type="span", **kw) as obs:
            yield obs

    def flush(self) -> None:
        if self.langfuse is not None:
            with contextlib.suppress(Exception):
                self.langfuse.flush()

    def shutdown(self) -> None:
        if self.langfuse is not None:
            with contextlib.suppress(Exception):
                self.langfuse.shutdown()
        if self.otel:
            with contextlib.suppress(Exception):
                from opentelemetry import trace

                trace.get_tracer_provider().shutdown()  # type: ignore[attr-defined]


NOOP = Observability()
