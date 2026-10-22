"""HTTP client for the NestJS backend. Used directly (REST transport) and by the MCP server."""

from typing import Any

import httpx


class ToolError(Exception):
    """A tool call failed; carries the backend's error envelope so the agent can explain it."""

    def __init__(self, code: str, message: str, details: Any = None, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details, "status": self.status}


class BackendClient:
    def __init__(self, base_url: str, timeout: float = 15.0, request_id: str | None = None):
        headers = {"x-request-id": request_id} if request_id else {}
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers=headers)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            res = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ToolError("BACKEND_UNREACHABLE", f"Backend not reachable: {exc}") from exc
        if res.status_code >= 400:
            try:
                err = res.json().get("error", {})
            except ValueError:
                err = {}
            raise ToolError(
                err.get("code", f"HTTP_{res.status_code}"),
                err.get("message", res.text[:300]),
                err.get("details"),
                res.status_code,
            )
        return res.json()

    def get_patient_profile(self, patient_id: str) -> dict[str, Any]:
        return self._request("GET", f"/patients/{patient_id}")

    def list_bookings(self, patient_id: str, when: str = "all") -> list[dict[str, Any]]:
        return self._request("GET", "/bookings", params={"patientId": patient_id, "when": when})

    def get_catalog(self) -> dict[str, Any]:
        return self._request("GET", "/catalog")

    def validate_booking(self, booking: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/bookings/validate", json=booking)

    def create_booking(self, booking: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/bookings", json=booking)

    def close(self) -> None:
        self._http.close()
