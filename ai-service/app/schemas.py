"""Typed contracts: what the LLM must return, and what the API returns."""

from typing import Any, Literal

from pydantic import BaseModel, Field

Intent = Literal[
    "clinic_info",  # opening hours, address, fees, policies, doctor details
    "recommendation",  # patient describes needs or symptoms and wants a doctor/treatment
    "booking_query",  # asks about their existing appointments
    "new_booking",  # wants to book (may include some details)
    "booking_update",  # supplies or changes details of the booking being drafted
    "confirm_booking",  # says yes to the booking summary
    "decline_booking",  # says no / wants to change something in the summary
    "abandon_booking",  # no longer wants to book
    "cancel_existing",  # wants to cancel or reschedule an existing appointment
    "emergency",  # describes an urgent or life-threatening situation
    "smalltalk",
    "out_of_scope",
]

INTENTS: tuple[str, ...] = Intent.__args__  # type: ignore[attr-defined]


class TurnAnalysis(BaseModel):
    """One structured LLM call per turn: intent classification + slot extraction."""

    intent: Intent = Field(description="The single best intent for the latest patient message.")
    confidence: float = Field(ge=0, le=1, description="Confidence in the intent, 0-1.")
    specialty: str | None = Field(
        None, description="Most relevant specialty from the allowed list, if the patient describes a need or symptom."
    )
    doctor: str | None = Field(None, description="Doctor id (e.g. D103) or name the patient refers to, if any.")
    treatment: str | None = Field(None, description="Treatment id (e.g. T003) or name the patient refers to, if any.")
    date: str | None = Field(None, description="Requested appointment date resolved to YYYY-MM-DD, if given.")
    time: str | None = Field(None, description="Requested appointment time as 24h HH:MM, if given.")
    notes: str | None = Field(None, description="Short reason for the visit or symptoms, in the patient's words.")


class MedicalSummary(BaseModel):
    """Bonus task: structured summary of a doctor-patient conversation."""

    chief_complaint: str = Field(description="Main reason for the visit, one sentence.")
    diagnosis: str = Field(description="Diagnosis stated by the doctor, or 'Not stated'.")
    medication: list[str] = Field(default_factory=list, description="Medications with dose/frequency if stated.")
    doctor_instructions: list[str] = Field(default_factory=list, description="Instructions and follow-up advice.")
    allergies_mentioned: list[str] = Field(default_factory=list, description="Allergies mentioned in the conversation.")
    follow_up: str | None = Field(None, description="Follow-up timing if stated.")


class TraceStep(BaseModel):
    step: str
    detail: str
    ms: int = 0


class ToolCall(BaseModel):
    name: str
    transport: str
    args: dict[str, Any]
    ok: bool
    ms: int


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    patient_id: str = Field(default="P001", pattern=r"^P\d{3,}$")


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    intent: str
    confidence: float
    booking_draft: dict[str, Any] | None
    awaiting_confirmation: bool
    booking_created: dict[str, Any] | None
    citations: list[str]
    tool_calls: list[ToolCall]
    trace: list[TraceStep]
    usage: dict[str, int]
    latency_ms: int


class SummarizeRequest(BaseModel):
    transcript: str = Field(min_length=20, max_length=20000)
