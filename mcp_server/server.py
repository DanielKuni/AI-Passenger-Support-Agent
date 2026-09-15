"""
MCP server for the AI Passenger Support Agent (DEMO).

Runs over stdio. Exposes two tools:
  - get_service_status(station)   -> simulated live status (reads demo_status.json on every call)
  - prepare_support_case(...)     -> appends a simulated support case to demo_cases.json

Everything returned here is DEMO data. Nothing is connected to a real operator system.
Run directly for a manual check:  python mcp_server/server.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

# The server runs as a separate process; make the shared redaction helper importable from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.redact import redact_mapping  # noqa: E402

# Hebrew in logs on Windows consoles: force UTF-8 on stderr (stdout is the MCP transport).
try:
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

HERE = Path(__file__).resolve().parent
STATUS_FILE = HERE / "demo_status.json"
CASES_FILE = HERE / "demo_cases.json"

server = MCPServer(
    name="redline-demo-ops",
    instructions="Simulated operational tools for the Tel Aviv Red Line passenger support demo. All data is demo data.",
)


def _log(msg: str) -> None:
    print(f"[mcp-server] {msg}", file=sys.stderr, flush=True)


def _load_status() -> dict:
    with STATUS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


@server.tool()
def get_service_status(station: str | None = None) -> dict:
    """Get the CURRENT (simulated) service status of the Tel Aviv Red Line light rail.

    Use this for any question that depends on the situation right now: delays, closed segments,
    closed stations, replacement buses, elevator/escalator availability at a station.
    Do not answer such questions from guidance documents.

    Args:
        station: Optional station name in Hebrew (e.g. "אלנבי"). When given, station-specific
                 information (elevators, notes) is included if available.
    """
    data = _load_status()
    scenario_name = data["active_scenario"]
    scenario = data["scenarios"][scenario_name]
    _log(f"get_service_status(station={station!r}) scenario={scenario_name}")

    if scenario.get("feed_error"):
        # Simulated outage: raise so the client receives an is_error result, not fake data.
        raise ToolError(scenario["feed_error"])

    station_info = None
    station_known = None
    if station:
        station_known = station in data["known_stations"]
        station_info = scenario.get("stations", {}).get(station)
        if station_known and station_info is None:
            station_info = {"elevators": "ok", "note_he": "אין דיווחים מיוחדים לתחנה זו."}
        # Is the station inside a closed segment?
        for d in scenario.get("disruptions", []):
            if station in d.get("affected_stations", []):
                station_info = dict(station_info or {})
                station_info["in_disrupted_segment"] = True

    return {
        "demo": True,
        "source": "simulated status feed (demo_status.json)",
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": scenario_name,
        "line_status": scenario["line_status"],
        "summary_he": scenario["summary_he"],
        "disruptions": scenario.get("disruptions", []),
        "station_query": station,
        "station_known": station_known,
        "station_info": station_info,
    }


CaseCategory = Literal[
    "payment_refund", "double_charge", "fine_appeal", "complaint",
    "lost_item", "accessibility", "disruption", "other",
]


@server.tool()
def prepare_support_case(
    category: CaseCategory,
    summary_he: str,
    passenger_details_he: str = "",
    checks_done_he: str = "",
) -> dict:
    """Prepare a (simulated) support case of the kind a HUMAN representative would handle.

    Call this when the passenger needs something the assistant cannot do or verify:
    refunds, fine appeals, complaints, injuries, conflicting or missing guidance.
    Include ONLY details the passenger actually gave. Never include full card numbers,
    passwords or ID numbers. The case is stored in the demo file only; it is NOT sent to
    any real service team and nobody will contact the passenger.

    Args:
        category: Case category.
        summary_he: Short Hebrew summary of the passenger's issue.
        passenger_details_he: Details the passenger provided (station, time, last 4 digits, etc.).
        checks_done_he: What the assistant already checked (e.g. service status result).
    """
    # Storage-side redaction: even a client that forgot to redact cannot write a secret to the demo file.
    fields, redaction_kinds = redact_mapping({
        "summary_he": summary_he, "passenger_details_he": passenger_details_he, "checks_done_he": checks_done_he,
    })
    cases: list[dict] = []
    if CASES_FILE.exists():
        with CASES_FILE.open("r", encoding="utf-8") as f:
            cases = json.load(f)
    case_id = f"DEMO-{len(cases) + 1:04d}"
    case = {
        "demo": True,
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "demo_only_not_sent",
        "note": "Demo data. Stored in demo_cases.json only; not sent to a real service team.",
        "category": category,
        "summary_he": fields["summary_he"],
        "passenger_details_he": fields["passenger_details_he"],
        "checks_done_he": fields["checks_done_he"],
        "redacted_on_store": sorted(set(redaction_kinds)),
    }
    cases.append(case)
    with CASES_FILE.open("w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    _log(f"prepare_support_case -> {case_id} ({category})")
    return case


if __name__ == "__main__":
    _log("starting on stdio")
    server.run("stdio")
