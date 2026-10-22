"""ClinicPilot MCP server.

Exposes the clinic tools over the Model Context Protocol (stdio). The agent uses it when
TOOL_TRANSPORT=mcp, and any MCP client (e.g. Claude Desktop, MCP Inspector) can use it too:

    python -m app.mcp_server

Every tool delegates to the backend REST API, so validation and persistence stay in one place.
Failures are returned as {"error": {...}} instead of raising, so clients get a structured reason.
"""

import os
from typing import Any

from mcp.server.mcpserver import MCPServer

from .backend_client import BackendClient, ToolError

server = MCPServer(
    name="clinicpilot",
    instructions="Clinic tools for patient P001's assistant: profile, bookings, catalog, validate and create bookings.",
)
_backend = BackendClient(os.environ.get("BACKEND_URL", "http://localhost:3000/api"))


def _safe(fn, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except ToolError as exc:
        return {"error": exc.to_dict()}


@server.tool(description="Get a patient's profile: demographics, allergies, bookings and visit history.")
def get_patient_profile(patient_id: str) -> dict[str, Any]:
    return _safe(_backend.get_patient_profile, patient_id)


@server.tool(description="List a patient's bookings. when = 'upcoming' | 'past' | 'all'.")
def list_bookings(patient_id: str, when: str = "all") -> list[dict[str, Any]] | dict[str, Any]:
    return _safe(_backend.list_bookings, patient_id, when)


@server.tool(description="Get all clinics, doctors and treatments (with the specialties that provide each treatment).")
def get_catalog() -> dict[str, Any]:
    return _safe(_backend.get_catalog)


@server.tool(description="Dry-run a booking against the clinic rules without saving it. Returns {valid, problems}.")
def validate_booking(booking: dict[str, Any]) -> dict[str, Any]:
    return _safe(_backend.validate_booking, booking)


@server.tool(
    description=(
        "Create a confirmed booking. Fields: patientId, doctorId, treatmentId, bookingDate (YYYY-MM-DD), "
        "bookingTime (HH:MM), notes, source, idempotencyKey."
    )
)
def create_booking(booking: dict[str, Any]) -> dict[str, Any]:
    return _safe(_backend.create_booking, booking)


if __name__ == "__main__":
    server.run(transport="stdio")
