"""MCP server contract test: connect an MCP client to the server in-process and call tools."""

import asyncio

from mcp import Client

import app.mcp_server as mcp_server


def test_mcp_server_lists_and_calls_tools(backend, monkeypatch):
    monkeypatch.setattr(mcp_server, "_backend", backend)

    async def run():
        async with Client(mcp_server.server) as client:
            tools = await client.list_tools()
            names = sorted(t.name for t in tools.tools)
            catalog = await client.call_tool("get_catalog", {})
            created = await client.call_tool(
                "create_booking",
                {"booking": {"patientId": "P001", "doctorId": "D104", "treatmentId": "T005",
                             "bookingDate": "2026-10-27", "bookingTime": "11:00", "source": "chatbot"}},
            )
            invalid = await client.call_tool(
                "create_booking",
                {"booking": {"patientId": "P001", "doctorId": "D103", "treatmentId": "T003",
                             "bookingDate": "2026-10-24", "bookingTime": "10:00"}},
            )
            return names, catalog, created, invalid

    names, catalog, created, invalid = asyncio.run(run())
    assert names == ["create_booking", "get_catalog", "get_patient_profile", "list_bookings", "validate_booking"]
    assert len(catalog.structured_content["doctors"]) == 4
    assert created.structured_content["booking"]["bookingId"] == "B1005"
    assert invalid.structured_content["error"]["code"] == "BOOKING_INVALID"
