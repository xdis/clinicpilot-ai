"""Observability tests (offline): the Langfuse trace of a turn is captured with an in-memory OpenTelemetry
exporter, so the structure is checked without a Langfuse server."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.agent import ClinicAgent
from app.fake_llm import FakeLLM
from app.knowledge import KnowledgeBase
from app.memory import PatientMemory
from app.telemetry import NOOP, Observability, mask_pii, pseudonym

from conftest import TODAY, FakeTools


def test_mask_pii_redacts_identifiers_but_keeps_dates_and_ids():
    out = mask_pii(data={"m": "Call +1-555-019-2831 or john.doe@mail.com, born 1988-05-14, booked 2026-10-28 at 10:00",
                         "ids": ["P001", "B1005"]})
    assert out["m"] == "Call [phone] or [email], born [date], booked 2026-10-28 at 10:00"
    assert out["ids"] == ["P001", "B1005"]


def test_patient_record_fields_are_masked_by_key():
    out = mask_pii(data={"profile": {"patientId": "P001", "fullName": "John Doe", "dob": "1988-05-14",
                                     "email": "j@d.com", "allergies": ["Peanuts"]}})
    assert out["profile"] == {"patientId": "P001", "fullName": "[redacted]", "dob": "[redacted]",
                              "email": "[redacted]", "allergies": ["Peanuts"]}


def test_pseudonym_is_stable_and_not_the_patient_id():
    assert pseudonym("P001") == pseudonym("P001") != pseudonym("P002")
    assert "P001" not in pseudonym("P001")


def test_noop_observability_changes_nothing(agent):
    assert NOOP.langfuse is None and NOOP.handler() == []
    r = agent.chat("s1", "P001", "What are the opening hours of Metro Care?")
    assert "trace_id" not in r


@pytest.fixture
def traced(backend):
    from langfuse import Langfuse

    exporter = InMemorySpanExporter()
    obs = Observability()
    import uuid

    obs._public_key = f"pk-test-{uuid.uuid4().hex}"  # Langfuse clients are singletons per public key
    obs.langfuse = Langfuse(public_key=obs._public_key, secret_key="sk-test", host="http://127.0.0.1:9", mask=mask_pii,
                            tracer_provider=TracerProvider(), span_exporter=exporter, flush_at=1, timeout=1)
    obs.scores = []
    obs.score = lambda trace_id, name, value, data_type="NUMERIC", comment=None: obs.scores.append((trace_id, name, value))
    memory = PatientMemory(InMemoryStore(), background=False)
    agent = ClinicAgent(llm=FakeLLM(), tools_factory=lambda: FakeTools(backend),
                        kb_loader=lambda tools: KnowledgeBase(tools.call("get_catalog")),
                        checkpointer=InMemorySaver(), clinic_today=TODAY, memory=memory, obs=obs)
    yield agent, obs, exporter
    obs.langfuse.shutdown()


def test_langfuse_down_never_fails_a_turn(traced):
    agent, _, _ = traced  # host points to a closed port
    r = agent.chat("sess-0", "P001", "What are the opening hours of Metro Care?")
    assert r["reply"] and r["trace_id"] and r["trace_url"] is None


def _spans(obs, exporter):
    obs.langfuse.flush()
    return exporter.get_finished_spans()


def test_turn_is_one_trace_with_graph_nodes_and_session(traced):
    agent, obs, exporter = traced
    r = agent.chat("sess-1", "P001", "What are the opening hours of Metro Care?")
    spans = _spans(obs, exporter)
    names = [s.name for s in spans]
    assert "clinicpilot.chat_turn" in names and "clinicpilot_graph" in names
    for node in ("load_context", "guard", "analyze", "retrieve", "answer"):
        assert node in names
    assert len({s.context.trace_id for s in spans}) == 1  # one turn, one trace
    assert format(spans[0].context.trace_id, "032x") == r["trace_id"]
    turn = next(s for s in spans if s.name == "clinicpilot.chat_turn")
    attrs = dict(turn.attributes or {})
    assert attrs.get("session.id") == "sess-1"
    assert attrs.get("user.id") == pseudonym("P001")  # never the raw patient id


def test_background_memory_write_joins_the_turn_trace(traced):
    agent, obs, exporter = traced
    r = agent.chat("sess-2", "P001", "I prefer morning appointments")
    spans = _spans(obs, exporter)
    bg = [s for s in spans if s.name == "memory.extract_preferences"]
    assert bg, [s.name for s in spans]
    assert format(bg[0].context.trace_id, "032x") == r["trace_id"]
    assert dict(bg[0].attributes or {}).get("session.id") == "sess-2"


def test_trace_payloads_are_masked(traced, backend):
    agent, obs, exporter = traced
    agent.chat("sess-3", "P001", "My email is john.doe@mail.com, what are the opening hours of Metro Care?")
    payload = " ".join(str(v) for s in _spans(obs, exporter) for v in (s.attributes or {}).values())
    assert "john.doe@mail.com" not in payload and "[email]" in payload


def test_online_scores_are_attached_to_the_turn(traced):
    agent, obs, _ = traced
    r = agent.chat("sess-4", "P001", "What are the opening hours of Metro Care?")
    got = {name: value for tid, name, value in obs.scores if tid == r["trace_id"]}
    assert set(got) == {"citations_valid", "guardrail", "intent_confidence", "latency_ms", "llm_calls", "tokens",
                        "memory_recalled"}
    assert got["citations_valid"] == 1 and got["guardrail"] == "none"
    agent.chat("sess-4", "P001", "I have chest pain and can't breathe")
    assert ("guardrail", "emergency") in [(n, v) for _, n, v in obs.scores]


def test_grounding_is_returned_only_in_evaluation_mode(backend, agent):
    assert "grounding" not in agent.chat("g0", "P001", "What are the opening hours of Metro Care?")
    eval_agent = ClinicAgent(llm=FakeLLM(), tools_factory=lambda: FakeTools(backend),
                             kb_loader=lambda tools: KnowledgeBase(tools.call("get_catalog")),
                             checkpointer=InMemorySaver(), clinic_today=TODAY, expose_grounding=True)
    r = eval_agent.chat("g1", "P001", "What are the opening hours of Metro Care?")
    assert any(f.startswith("[C001]") for f in r["grounding"])  # the exact facts the answer was written from
    assert r["grounding"][0].startswith("ASSISTANT RULES") and "CONVERSATION" in r["grounding"][-1]
    r = eval_agent.chat("g1", "P001", "I want to book an appointment with Dr Rostova")
    assert r["grounding"] == []  # booking replies are templates: nothing stale from the previous turn


def test_doctor_notes_from_the_visit_history_reach_the_answer(backend):
    eval_agent = ClinicAgent(llm=FakeLLM(), tools_factory=lambda: FakeTools(backend),
                             kb_loader=lambda tools: KnowledgeBase(tools.call("get_catalog")),
                             checkpointer=InMemorySaver(), clinic_today=TODAY, expose_grounding=True)
    r = eval_agent.chat("h1", "P001", "What did the doctor say at my last checkup?")
    record = next(f for f in r["grounding"] if f.startswith("[P001]"))
    assert "doctor's notes: Blood pressure normal" in record and "reason: Annual routine health checkup" in record
    assert "never recommend, renew or change medication" in record
