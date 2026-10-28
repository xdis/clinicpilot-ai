"""Patient memory: short-term (within a session) and long-term (across sessions).

Layers (see TECH_DOC "Memory"):
  0. Facts       - allergies, bookings, visit history: read from the backend every turn, never taken from memory.
  1. Short-term  - LangGraph checkpointer state per thread (patient:session): structured booking draft,
                   recent turns kept inside a token budget, and a rolling summary of older turns.
  2. Long-term   - LangGraph BaseStore (SqliteStore here, PostgresStore + pgvector in production),
                   namespaced per (tenant, patient):
                     ("profile")    one document of validated preferences (language, time of day, doctor, clinic)
                     ("episodes")   short summaries of past conversations, recalled by term overlap + recency
                     ("candidates") safety-relevant statements (e.g. a new allergy) parked for staff review,
                                    never shown to the model
                     ("audit")      every write and delete, with provenance
Writes happen off the hot path (a single background worker), are gated by cheap rules, and are validated
in code: whitelisted fields, ids that exist in the catalog, no instructions, no safety facts.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from .safety import looks_like_injection

log = logging.getLogger(__name__)

TimeOfDay = Literal["morning", "afternoon", "evening"]
LANGUAGES = {"english", "cantonese", "mandarin", "chinese", "spanish", "french", "german", "hindi", "tagalog", "japanese", "korean"}
# Only enumerated, validated fields: no free text in the long-term profile (free text is an injection channel).
PROFILE_FIELDS = ("preferred_language", "preferred_time_of_day", "preferred_doctor_id", "preferred_clinic_id")

# Cheap gate: only messages that look like they state a lasting preference (or a correction) trigger extraction.
_PREFERENCE_CUE = re.compile(
    r"\b(prefer|preference|usually|always|normally|only|rather|instead|from now on|next time|in future|in the future|"
    r"mornings?|afternoons?|evenings?|before work|after work|speak|language|in (english|cantonese|mandarin|chinese|spanish)|"
    r"call me|my usual|my regular|don'?t like|do not like|no longer|not anymore|changed? my)\b"
    r"|(喜欢|喜歡|习惯|習慣|通常|上午|下午|晚上|广东话|廣東話|普通话|普通話|粤语|粵語)",
    re.I,
)
# Safety-relevant statements are facts for the medical record, not preferences: park them for staff review.
_SAFETY = re.compile(
    r"\b(allerg\w*|anaphyla\w*|medication|medicine|mg\b|dose|dosage|prescri\w*|pregnan\w*|diabet\w*|"
    r"blood thinner|warfarin|insulin|epipen|asthma|condition|diagnos\w*)\b|(过敏|過敏|药|藥|怀孕|懷孕)",
    re.I,
)
_WORD = re.compile(r"[a-z0-9]+|[一-鿿]")
# A preference must be the patient talking about themselves ("I prefer...", "speak Cantonese with me").
_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|me|my|mine)\b|我", re.I)
# ...and not about someone else ("which doctors speak Spanish for my mother?").
_THIRD_PARTY = re.compile(
    r"\b(my|our|for)\s+(mother|mom|mum|father|dad|parents?|wife|husband|son|daughter|child|children|kids?|baby|"
    r"partner|friend|brother|sister|grand\w+)\b|\bfor (him|her|them)\b|(我妈|我媽|我爸|我老婆|我老公|我儿子|我兒子|我女儿|我女兒|我孩子)",
    re.I,
)
# Words that name a clinic generically; a reference made only of these identifies nothing.
_GENERIC_CLINIC = {"clinic", "center", "centre", "health", "care", "medical", "wellness", "the", "a", "of"}
_STOP = {"a", "an", "the", "i", "my", "me", "is", "am", "are", "to", "of", "and", "or", "in", "on", "for", "about",
         "again", "it", "do", "does", "can", "what", "who", "how", "when", "should", "with", "this", "that", "you"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def to_messages(history: list[dict[str, str]]) -> list[BaseMessage]:
    return [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in history]


def from_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
    return [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": str(m.content)} for m in messages]


def redact_injections(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """User text that looks like instructions is not carried into summaries (which reach the system prompt)."""
    return [{"role": m["role"], "content": "[message removed: looked like instructions to the assistant]"}
            if m["role"] == "user" and looks_like_injection(m["content"]) else m for m in messages]


# ---------------------------------------------------------------------------------------------- schemas
class PreferenceUpdate(BaseModel):
    """What the extractor may propose. Code validates every field before anything is stored."""

    preferred_language: str | None = Field(None, description="Language the patient asks to be served in, e.g. Cantonese.")
    preferred_time_of_day: TimeOfDay | None = Field(None, description="morning / afternoon / evening, only if stated as a habit.")
    preferred_doctor: str | None = Field(None, description="Doctor id or name the patient says they prefer or usually see.")
    preferred_clinic: str | None = Field(None, description="Clinic id or name the patient prefers.")
    clear_fields: list[str] = Field(default_factory=list, description="Profile fields the patient withdrew, e.g. preferred_doctor.")
    safety_statement: str | None = Field(
        None, description="A new allergy, medication or condition the patient mentioned, verbatim. It is NOT stored as memory."
    )


class EpisodeSummary(BaseModel):
    """One past conversation, in two sentences, without diagnoses or medication."""

    summary: str = Field(description="What the patient wanted and what happened (booked, recommended, unresolved). Max 2 sentences.")
    topics: list[str] = Field(default_factory=list, description="Up to 5 short keywords, e.g. knee pain, orthopedics, price.")
    open_item: str | None = Field(None, description="Anything left unresolved that a later conversation may pick up.")


# ---------------------------------------------------------------------------------------------- short-term
class ShortTermMemory:
    """Token-budgeted window over the session history plus a rolling summary of what fell out of it.

    Invariant: every message of the conversation is either in the window or in the summary (no gap).
    - The structured booking draft lives in its own state fields, so it is never trimmed or summarised.
    - `trim_messages(strategy="last", start_on="human")` keeps whole turns, newest first.
    - High/low watermark: as soon as the history no longer fits the budget (`max_tokens`) or the turn cap
      (`max_messages`), the oldest part is folded into `summary` with one LLM call until only `keep_tokens`
      (half the budget by default) remain. Folded messages are dropped from the checkpoint, so state stays
      bounded, and the freed half-budget means the next few turns need no summarisation call.
    """

    def __init__(self, max_tokens: int = 1200, keep_tokens: int | None = None, max_messages: int = 16,
                 token_counter: Any = "approximate"):
        self.max_tokens = max_tokens
        self.keep_tokens = keep_tokens or max_tokens // 2
        self.max_messages = max_messages
        self.token_counter = token_counter

    def count(self, history: list[dict[str, str]]) -> int:
        if not history:
            return 0
        msgs = to_messages(history)
        if self.token_counter == "approximate":
            from langchain_core.messages.utils import count_tokens_approximately

            return count_tokens_approximately(msgs)
        return self.token_counter(msgs)

    def _trim(self, history: list[dict[str, str]], budget: int, max_messages: int) -> list[dict[str, str]]:
        if not history:
            return []
        kept = from_messages(trim_messages(
            to_messages(history[-max_messages:]), max_tokens=budget, token_counter=self.token_counter,
            strategy="last", start_on="human", allow_partial=False,
        ))
        if not kept:  # the newest turn alone is over budget: keep it anyway rather than lose all context
            users = [i for i, m in enumerate(history) if m["role"] == "user"]
            kept = history[users[-1]:] if users else history[-1:]
        return kept

    def window(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        """What the model sees: newest whole turns within the budget and the turn cap."""
        return self._trim(history, self.max_tokens, self.max_messages)

    def compact(self, history: list[dict[str, str]], summary: str,
                summarize: Callable[[str, list[dict[str, str]]], str]) -> tuple[list[dict[str, str]], str, int]:
        """Folds the oldest messages into the rolling summary when the history overflows the window.

        Returns (history, summary, folded). On a summarisation error the raw history is kept (retried next turn).
        """
        if len(self.window(history)) == len(history):
            return history, summary, 0
        keep = self._trim(history, self.keep_tokens, self.max_messages // 2)
        old = history[: len(history) - len(keep)]
        try:
            new_summary = summarize(summary, redact_injections(old))
        except Exception:  # noqa: BLE001 - keep the raw turns rather than lose them
            log.warning("rolling summary failed; keeping raw history", exc_info=True)
            return history, summary, 0
        return keep, new_summary[:1500], len(old)


# ---------------------------------------------------------------------------------------------- long-term
class PatientMemory:
    """Long-term memory on a LangGraph BaseStore, isolated per (tenant, patient)."""

    def __init__(self, store: BaseStore, *, tenant: str = "default", background: bool = True,
                 episode_limit: int = 50, retention_days: int = 365):
        self.store = store
        self.tenant = tenant
        self.episode_limit = episode_limit
        self.retention_days = retention_days
        self._lock = threading.Lock()  # one writer at a time (the store connection is shared)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory") if background else None
        self._pending: list[Future] = []
        self._erased_at: dict[str, float] = {}  # patient -> time of last erasure; older queued writes are dropped

    # namespaces ---------------------------------------------------------------------------
    def ns(self, patient_id: str, kind: str) -> tuple[str, ...]:
        return ("tenant", self.tenant, "patient", patient_id, kind)

    # background execution -----------------------------------------------------------------
    def submit(self, fn: Callable[..., Any], *args: Any) -> None:
        """Runs a write off the hot path (or inline when background=False, e.g. tests)."""
        if self._pool is None:
            fn(*args)
            return
        fut = self._pool.submit(self._safe, fn, *args)
        self._pending = [f for f in self._pending if not f.done()] + [fut]

    @staticmethod
    def _safe(fn: Callable[..., Any], *args: Any) -> None:
        try:
            fn(*args)
        except Exception:  # noqa: BLE001 - a failed memory write must never affect the conversation
            log.exception("memory write failed")

    def flush(self, timeout: float = 30.0) -> None:
        for f in list(self._pending):
            f.result(timeout=timeout)
        self._pending = []

    # read path (hot) ----------------------------------------------------------------------
    def profile(self, patient_id: str) -> dict[str, Any]:
        item = self.store.get(self.ns(patient_id, "profile"), "preferences")
        return dict(item.value) if item else {}

    def episodes(self, patient_id: str, limit: int = 100) -> list[dict[str, Any]]:
        items = self.store.search(self.ns(patient_id, "episodes"), limit=limit)
        cutoff = time.time() - self.retention_days * 86400
        eps = [dict(i.value, id=i.key) for i in items if i.value.get("ts", 0) >= cutoff]
        return sorted(eps, key=lambda e: e.get("ts", 0), reverse=True)

    def recall(self, patient_id: str, query: str, k: int = 3) -> dict[str, Any]:
        """Profile + the k most relevant episodes: lexical relevance blended with recency.

        Per-patient collections are tiny (tens of items). Okapi BM25's IDF, log((N - df + 0.5) / (df + 0.5)),
        is zero or negative once a term is in half the items, so terms are weighted with log(1 + N/df) instead.
        Semantic recall in production: give the store an embedding index (PostgresStore + pgvector), write
        episodes with index=["summary"] instead of index=False, and call store.search(ns, query=...).
        """
        eps = self.episodes(patient_id)
        top: list[dict[str, Any]] = []
        if eps:
            docs = [set(_tokens(f"{e['summary']} {' '.join(e.get('topics', []))} {e.get('open_item') or ''}")) for e in eps]
            q = set(_tokens(query)) - _STOP
            df = {t: sum(t in d for d in docs) for t in q}
            scores = [sum(math.log(1 + len(docs) / df[t]) for t in q if t in d) for d in docs]
            hi = max(scores) if scores else 0.0
            ranked = []
            for rank, (e, sc) in enumerate(zip(eps, scores)):  # eps are newest first
                recency = 1.0 / (1 + rank)
                ranked.append((0.7 * (sc / hi if hi > 0 else 0) + 0.3 * recency, sc, e))
            ranked.sort(key=lambda t: t[0], reverse=True)
            # relevant episodes only; with no term overlap at all, just the most recent one for continuity
            top = [e for _, sc, e in ranked if sc > 0][:k] if hi > 0 else eps[:1]
        return {"profile": self.profile(patient_id), "episodes": top}

    @staticmethod
    def prompt_block(recalled: dict[str, Any], kb: Any) -> list[str]:
        """Memory rendered as DATA for the prompt, clearly marked as lower priority than database facts."""
        lines: list[str] = []
        p = recalled.get("profile") or {}
        prefs = []
        if p.get("preferred_language"):
            prefs.append(f"prefers to be served in {p['preferred_language']['value']}")
        if p.get("preferred_time_of_day"):
            prefs.append(f"prefers {p['preferred_time_of_day']['value']} appointments")
        if p.get("preferred_doctor_id"):
            d = kb.doctors.get(p["preferred_doctor_id"]["value"])
            if d:
                prefs.append(f"usually sees {d['fullName']} [{d['id']}]")
        if p.get("preferred_clinic_id"):
            c = kb.clinics.get(p["preferred_clinic_id"]["value"])
            if c:
                prefs.append(f"prefers {c['name']} [{c['id']}]")
        if prefs:
            lines.append("Patient preferences remembered from earlier conversations (may be outdated; the patient's "
                         "latest message and the database always win): " + "; ".join(prefs) + ".")
        for e in recalled.get("episodes", []):
            day = e.get("date", "")
            extra = f" Open item: {e['open_item']}" if e.get("open_item") else ""
            lines.append(f"Earlier conversation ({day}): {e['summary']}{extra}")
        return lines

    # write path (cold) --------------------------------------------------------------------
    def should_extract(self, message: str, flags: list[str]) -> bool:
        if "possible_prompt_injection" in flags:
            return False
        return bool(_PREFERENCE_CUE.search(message) or _SAFETY.search(message))

    def apply_preferences(self, patient_id: str, update: PreferenceUpdate, *, kb: Any, message: str,
                          session_id: str, extractor: str, since: float | None = None) -> list[dict[str, Any]]:
        """Validates the extractor's proposal and writes what survives. Returns the audit entries."""
        changes: dict[str, Any] = {}
        rejected: list[str] = []
        if not _FIRST_PERSON.search(message) or _THIRD_PARTY.search(message):
            # "Which doctors speak Spanish?" is a question; "...for my mother" is about someone else
            proposed = update.model_dump(exclude={"safety_statement"}, exclude_none=True, exclude_defaults=True)
            if proposed:
                rejected.append(f"{sorted(proposed)} (message is not the patient stating their own preference)")
            update = PreferenceUpdate(safety_statement=update.safety_statement)
        if update.preferred_language:
            lang = update.preferred_language.strip().lower()
            if lang in LANGUAGES:
                changes["preferred_language"] = lang.title()
            else:
                rejected.append(f"preferred_language={update.preferred_language!r} (not a supported language)")
        if update.preferred_time_of_day:
            changes["preferred_time_of_day"] = update.preferred_time_of_day
        if update.preferred_doctor:
            doc = self._named_doctor(update.preferred_doctor, kb)
            if doc:
                changes["preferred_doctor_id"] = doc["id"]
            else:
                rejected.append(f"preferred_doctor={update.preferred_doctor!r} (not a doctor named in the catalog)")
        if update.preferred_clinic:
            clinic = self._named_clinic(update.preferred_clinic, kb)
            if clinic:
                changes["preferred_clinic_id"] = clinic["id"]
            else:
                rejected.append(f"preferred_clinic={update.preferred_clinic!r} (no such clinic)")
        clear = {f if f in PROFILE_FIELDS else f"{f}_id" for f in update.clear_fields}
        clear = {f for f in clear if f in PROFILE_FIELDS}

        entries: list[dict[str, Any]] = []
        source = {"session_id": session_id, "message_sha256": hashlib.sha256(message.encode()).hexdigest()[:16],
                  "extractor": extractor, "at": _now()}
        with self._lock:
            if self._erased_since(patient_id, since):
                return []
            ns = self.ns(patient_id, "profile")
            item = self.store.get(ns, "preferences")
            profile = dict(item.value) if item else {}
            for field in clear:
                if field in profile:
                    entries.append(self._audit(patient_id, "clear", field, profile[field]["value"], None, source))
                    profile.pop(field)
            for field, value in changes.items():
                before = profile.get(field, {}).get("value")
                if before == value:
                    continue
                profile[field] = {"value": value, "source": source, "previous": before}
                entries.append(self._audit(patient_id, "set", field, before, value, source))
            if entries:
                self.store.put(ns, "preferences", profile, index=False)
            # only from the latest message (the extractor also sees earlier turns), and only once per statement
            if update.safety_statement and _SAFETY.search(message):
                stmt = update.safety_statement.strip()[:300]
                existing = {c.value["statement"].strip().lower()
                            for c in self.store.search(self.ns(patient_id, "candidates"), limit=1000)}
                if stmt.lower() not in existing:
                    self.store.put(self.ns(patient_id, "candidates"), uuid.uuid4().hex,
                                   {"statement": stmt, "status": "pending_staff_review", "source": source}, index=False)
                    entries.append(self._audit(patient_id, "candidate", "safety_statement", None, "pending_staff_review", source))
            for r in rejected:
                entries.append(self._audit(patient_id, "rejected", r, None, None, source))
        return entries

    @staticmethod
    def _named_doctor(ref: str, kb: Any) -> dict[str, Any] | None:
        """A doctor id or surname, never a specialty ("the dermatologist" is not a preference for one person)."""
        r = ref.strip().lower()
        for d in kb.doctors.values():
            if r == d["id"].lower() or re.search(rf"\b{re.escape(d['fullName'].split()[-1].lower())}\b", r):
                return d
        return None

    @staticmethod
    def _named_clinic(ref: str, kb: Any) -> dict[str, Any] | None:
        """A clinic id, or distinctive words of exactly one clinic's name ("Metro Care" yes; "care" or " " no)."""
        r = ref.strip().lower()
        words = [w for w in _tokens(r) if w not in _GENERIC_CLINIC]
        if not words:
            return None
        hits = [c for c in kb.clinics.values()
                if r == c["id"].lower() or all(w in _tokens(c["name"]) for w in words)]
        return hits[0] if len(hits) == 1 else None

    @staticmethod
    def _clean(text: str | None, limit: int) -> str | None:
        """Episode text that may reach a prompt: drop sentences with medical facts or instruction-like text."""
        if not text:
            return None
        sentences = re.split(r"(?<=[.!?。！？])\s+", re.sub(r"\s+", " ", text).strip())
        # per sentence: "the patient tried to make the assistant book for P002" must not cost the whole episode
        kept = [x for x in sentences if x and not _SAFETY.search(x) and not looks_like_injection(x)]
        out = " ".join(kept).strip()[:limit]
        if not out or looks_like_injection(out):
            return None
        return out

    def add_episode(self, patient_id: str, ep: EpisodeSummary, *, session_id: str, extractor: str, day: str,
                    conversation_id: str = "", since: float | None = None) -> dict[str, Any] | None:
        summary = self._clean(ep.summary, 300)
        if not summary:
            log.warning("episode for session %s dropped: nothing left after the safety filter", session_id)
            return None
        value = {
            "summary": summary,
            "topics": [t for t in (self._clean(x, 30) for x in ep.topics[:5]) if t],
            "open_item": self._clean(ep.open_item, 160),
            "date": day,
            "ts": time.time(),
            "source": {"session_id": session_id, "extractor": extractor, "at": _now()},
        }
        with self._lock:
            if self._erased_since(patient_id, since):
                return None
            # one episode per conversation: ending it twice replaces it; a session id reused after reset does not
            key = f"{session_id}-{conversation_id}" if conversation_id else session_id
            self.store.put(self.ns(patient_id, "episodes"), key, value, index=False)
            self._prune(patient_id)
            return self._audit(patient_id, "episode", key, None, value["summary"], value["source"])

    def _erased_since(self, patient_id: str, since: float | None) -> bool:
        return since is not None and self._erased_at.get(patient_id, 0.0) >= since

    def _prune(self, patient_id: str) -> None:
        """Retention on disk, not just on read: delete expired episodes and keep the newest N."""
        ns = self.ns(patient_id, "episodes")
        items = sorted(self.store.search(ns, limit=10_000), key=lambda i: i.value.get("ts", 0), reverse=True)
        cutoff = time.time() - self.retention_days * 86400
        for rank, i in enumerate(items):
            if rank >= self.episode_limit or i.value.get("ts", 0) < cutoff:
                self.store.delete(ns, i.key)

    def forget(self, patient_id: str) -> dict[str, int]:
        """Right to erasure: deletes every memory item for the patient. The audit keeps only a hash of the id.

        Writes already queued for this patient are dropped (tombstone), so erasure never waits for the LLM.
        """
        counts: dict[str, int] = {}
        with self._lock:
            self._erased_at[patient_id] = time.time()
            for kind in ("profile", "episodes", "candidates", "audit"):
                items = self.store.search(self.ns(patient_id, kind), limit=10_000)
                for i in items:
                    self.store.delete(self.ns(patient_id, kind), i.key)
                counts[kind] = len(items)
            pid_hash = hashlib.sha256(f"{self.tenant}:{patient_id}".encode()).hexdigest()[:16]
            self.store.put(("tenant", self.tenant, "erasures"), uuid.uuid4().hex,
                           {"patient_sha256": pid_hash, "at": _now(), "counts": counts}, index=False)
        return counts

    def snapshot(self, patient_id: str) -> dict[str, Any]:
        audit = self.store.search(self.ns(patient_id, "audit"), limit=200)
        cands = self.store.search(self.ns(patient_id, "candidates"), limit=100)
        return {
            "namespace": list(self.ns(patient_id, "*")),
            "profile": self.profile(patient_id),
            "episodes": self.episodes(patient_id),
            "candidates": [dict(c.value, id=c.key) for c in cands],
            "audit": sorted((a.value for a in audit), key=lambda a: a["at"], reverse=True),
        }

    def _audit(self, patient_id: str, op: str, field: str, before: Any, after: Any, source: dict[str, Any]) -> dict[str, Any]:
        entry = {"op": op, "field": field, "before": before, "after": after, "at": _now(),
                 "session_id": source.get("session_id"), "extractor": source.get("extractor")}
        self.store.put(self.ns(patient_id, "audit"), uuid.uuid4().hex, entry, index=False)
        return entry
