"""Deterministic, offline stand-in for the LLM (LLM_PROVIDER=fake).

Used by the test suite and CI so the agent's routing, booking flow, guardrails and persistence
can be verified without network access or API cost. It is intentionally rule-based and narrow;
it is not meant to be a good assistant.
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

from .schemas import MedicalSummary, TurnAnalysis

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_SYMPTOMS = {
    "Dermatology": ["rash", "skin", "acne", "eczema", "itch"],
    "Orthopedics": ["knee", "joint", "back pain", "sprain", "sports injury"],
    "Pediatrics": ["child", "kid", "son", "daughter", "baby", "vaccin"],
    "General Practice": ["checkup", "check-up", "flu", "fever", "cough", "allergy", "allergic"],
}


class FakeLLM:
    name = "fake:rules"

    def __init__(self) -> None:
        self.usage = {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}

    def analyze(self, *, message: str, history: list[dict[str, str]], context: dict[str, Any]) -> TurnAnalysis:
        self.usage["llm_calls"] += 1
        m = message.lower()
        today = context["today"]
        draft = context["draft"]

        specialty = next((s for s, words in _SYMPTOMS.items() if any(w in m for w in words)), None)
        doctor = None
        for line in context["doctor_lines"]:
            did, rest = line[2:].split(": ", 1)
            name = rest.split(" (")[0]
            last = name.split()[-1].lower()
            if did.lower() in m or last in m:
                doctor = did
        if doctor is None:
            named = re.search(r"\bdr\.? ([a-z]+)", m)
            if named:
                doctor = "Dr. " + named.group(1).title()  # unknown names are passed through, like a real model would
        if doctor is None and re.search(r"\b(her|him|that doctor)\b", m):
            for h in reversed(history):
                found = re.findall(r"\[(D\d{3})\]", h["content"])
                if found:
                    doctor = found[0]
                    break
        treatment = None
        for line in context["treatment_lines"]:
            tid, rest = line[2:].split(": ", 1)
            if tid.lower() in m or rest.split(" (")[0].lower() in m:
                treatment = tid

        date = None
        iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", m)
        if iso:
            date = iso.group(1)
        elif "tomorrow" in m:
            date = (today + timedelta(days=1)).isoformat()
        else:
            for i, wd in enumerate(_WEEKDAYS):
                if wd in m:
                    delta = (i - today.weekday()) % 7 or 7
                    date = (today + timedelta(days=delta)).isoformat()
        time = None
        tm = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", m) or re.search(r"\b(\d{1,2}):(\d{2})\b", m)
        if tm:
            h = int(tm.group(1))
            minute = tm.group(2) or "00"
            ampm = tm.group(3) if tm.lastindex and tm.lastindex >= 3 else None
            if ampm == "pm" and h < 12:
                h += 12
            if ampm == "am" and h == 12:
                h = 0
            time = f"{h:02d}:{minute}"

        if any(w in m for w in ["my appointment", "my booking", "next appointment", "my appointments", "upcoming"]):
            intent = "booking_query"
        elif re.search(r"\b(cancel|reschedule)\b", m) and "booking request" not in m:
            intent = "cancel_existing"
        elif re.search(r"\b(never ?mind|forget it|stop booking)\b", m):
            intent = "abandon_booking"
        elif re.search(r"\b(book|appointment|schedule)\b", m):
            intent = "new_booking" if not draft else "booking_update"
        elif draft and (date or time or doctor or treatment):
            intent = "booking_update"
        elif specialty:
            intent = "recommendation"
        elif re.search(r"\b(hours|open|address|fee|cost|price|policy|phone|speak|language)\b", m):
            intent = "clinic_info"
        elif re.search(r"\b(hi|hello|thanks|thank you)\b", m):
            intent = "smalltalk"
        else:
            intent = "out_of_scope"
        return TurnAnalysis(
            intent=intent, confidence=0.9, specialty=specialty, doctor=doctor, treatment=treatment, date=date,
            time=time, notes=message[:120] if specialty else None,
        )

    def compose(self, *, system: str, facts: str, history: list[dict[str, str]], message: str) -> str:
        self.usage["llm_calls"] += 1
        first = [line for line in facts.splitlines() if line.startswith("[")][:3]
        return "Here is what I found:\n" + "\n".join(f"- {line[:160]}" for line in first)

    def summarize(self, transcript: str) -> MedicalSummary:
        self.usage["llm_calls"] += 1
        return MedicalSummary(chief_complaint=transcript.splitlines()[0][:120], diagnosis="Not stated")
