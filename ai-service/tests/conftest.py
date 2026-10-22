"""Offline test fixtures: an in-memory backend that mirrors the real API contract, the rule-based
FakeLLM, and an in-memory LangGraph checkpointer. No network, no API key, deterministic."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import ClinicAgent
from app.backend_client import ToolError
from app.fake_llm import FakeLLM
from app.knowledge import KnowledgeBase
from app.tools import _Recorder

DATA = Path(__file__).resolve().parents[2] / "mock-data"
CATEGORY_TO_SPECIALTIES = {
    "Preventative Care": ["General Practice"],
    "Pediatrics": ["Pediatrics"],
    "Dermatology": ["Dermatology"],
    "Immunization": ["General Practice", "Pediatrics"],
    "Orthopedics": ["Orthopedics"],
}
CLOSED = {"C001": {"Sun"}, "C002": {"Sat", "Sun"}}
TODAY = "2026-10-22"


def _rows(name: str) -> list[dict[str, str]]:
    with open(DATA / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class FakeBackend:
    def __init__(self) -> None:
        self.clinics = [
            {"id": r["clinic_id"], "name": r["clinic_name"], "address": r["address"], "phone": r["phone"],
             "operatingHours": r["operating_hours"], "cancellationPolicy": r["cancellation_policy"],
             "emergencyInstructions": r["emergency_instructions"]}
            for r in _rows("clinics.csv")
        ]
        self.doctors = [
            {"id": r["doctor_id"], "clinicId": r["clinic_id"], "fullName": r["full_name"], "specialty": r["specialty"],
             "languagesSpoken": r["languages_spoken"], "consultationFeeUsd": float(r["consultation_fee_usd"]), "bio": r["bio"]}
            for r in _rows("doctors.csv")
        ]
        self.treatments = [
            {"id": r["treatment_id"], "name": r["treatment_name"], "category": r["category"],
             "durationMinutes": int(r["duration_minutes"]), "priceUsd": float(r["price_usd"]),
             "preTreatmentInstructions": r["pre_treatment_instructions"],
             "providedBySpecialties": CATEGORY_TO_SPECIALTIES[r["category"]]}
            for r in _rows("treatments.csv")
        ]
        self.bookings = [dict(r) for r in _rows("patient_bookings.csv")]
        self.created: list[dict[str, Any]] = []
        self.keys: dict[str, dict[str, Any]] = {}

    def _view(self, b: dict[str, Any]) -> dict[str, Any]:
        d = next(x for x in self.doctors if x["id"] == b["doctor_id"])
        t = next(x for x in self.treatments if x["id"] == b["treatment_id"])
        c = next(x for x in self.clinics if x["id"] == d["clinicId"])
        return {"bookingId": b["booking_id"], "patientId": b["patient_id"], "doctorId": d["id"], "doctorName": d["fullName"],
                "treatmentId": t["id"], "treatmentName": t["name"], "clinicId": c["id"], "clinicName": c["name"],
                "bookingDate": b["booking_date"], "bookingTime": b["booking_time"], "status": b["status"], "notes": b["notes"],
                "preTreatmentInstructions": t["preTreatmentInstructions"], "source": b.get("source", "seed")}

    def get_patient_profile(self, patient_id: str) -> dict[str, Any]:
        return {"patientId": patient_id, "fullName": "John Doe", "firstName": "John", "dob": "1988-05-14",
                "allergies": ["Penicillin", "Peanuts"], "bookings": self.list_bookings(patient_id),
                "treatmentHistory": [{"visitDate": "2026-04-12", "diagnosis": "Healthy / Normal Range", "doctorName": "Dr. Sarah Jenkins"}]}

    def list_bookings(self, patient_id: str, when: str = "all") -> list[dict[str, Any]]:
        return [self._view(b) for b in sorted(self.bookings, key=lambda b: b["booking_date"]) if b["patient_id"] == patient_id]

    def get_catalog(self) -> dict[str, Any]:
        return {"clinics": self.clinics, "doctors": self.doctors, "treatments": self.treatments}

    def validate_booking(self, booking: dict[str, Any]) -> dict[str, Any]:
        from datetime import date

        problems = []
        d = next(x for x in self.doctors if x["id"] == booking["doctorId"])
        t = next(x for x in self.treatments if x["id"] == booking["treatmentId"])
        if d["specialty"] not in t["providedBySpecialties"]:
            problems.append("specialty mismatch")
        if booking["bookingDate"] < TODAY:
            problems.append("date in the past")
        if date.fromisoformat(booking["bookingDate"]).strftime("%a") in CLOSED[d["clinicId"]]:
            problems.append("The clinic is closed that day.")
        return {"valid": not problems, "problems": problems}

    def create_booking(self, booking: dict[str, Any]) -> dict[str, Any]:
        key = booking.get("idempotencyKey")
        if key and key in self.keys:
            return {"booking": self.keys[key], "idempotentReplay": True}
        if not self.validate_booking(booking)["valid"]:
            raise ToolError("BOOKING_INVALID", "invalid", status=422)
        row = {"booking_id": f"B{1005 + len(self.created)}", "patient_id": booking["patientId"], "doctor_id": booking["doctorId"],
               "treatment_id": booking["treatmentId"], "booking_date": booking["bookingDate"],
               "booking_time": booking["bookingTime"] + ":00", "status": "Confirmed", "notes": booking.get("notes", ""), "source": "chatbot"}
        self.bookings.append(row)
        view = self._view(row)
        self.created.append(booking)
        if key:
            self.keys[key] = view
        return {"booking": view, "idempotentReplay": False}


class FakeTools(_Recorder):
    transport = "fake"

    def __init__(self, backend: FakeBackend):
        super().__init__()
        self.backend = backend

    def _invoke(self, name: str, args: dict[str, Any]) -> Any:
        return getattr(self.backend, name)(**args)


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def agent(backend: FakeBackend) -> ClinicAgent:
    return ClinicAgent(
        llm=FakeLLM(),
        tools_factory=lambda: FakeTools(backend),
        kb_loader=lambda tools: KnowledgeBase(tools.call("get_catalog")),
        checkpointer=InMemorySaver(),
        clinic_today=TODAY,
    )
