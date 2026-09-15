"""
Privacy test for demo handoffs: sensitive details typed by a passenger must not reach the stored demo case,
the MCP call input, the MCP call log, or the visible reply, while the useful description is kept.

The test runs the real deterministic workflow against the real MCP server, then opens
mcp_server/demo_cases.json and inspects the case that was actually written.

Run:  python -m tests.test_redaction
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app import config  # noqa: E402
from app.mcp_client import MCPBridge  # noqa: E402
from app.offline_workflow import run_free_question  # noqa: E402
from app.rag import BM25Retriever, load_passages  # noqa: E402
from app.redact import MARKER, redact_sensitive  # noqa: E402

# (question, secrets that must disappear everywhere, useful words that must survive in the stored summary)
CASES = [
    ("הסיסמה שלי לאפליקציה היא 1234, תבדוק למה חויבתי פעמיים", ["1234"], ["חויבתי פעמיים"]),
    ("password: Tr0ub4dor, ואני רוצה החזר על החיוב הכפול", ["Tr0ub4dor"], ["החזר"]),
    ("מספר תעודת הזהות שלי 012345678 ואני רוצה לערער על הקנס", ["012345678"], ["לערער", "קנס"]),
    ("הכרטיס שלי 4580 1234 1234 1234 חויב פעמיים אתמול", ["4580 1234 1234 1234", "4580123412341234"], ["פעמיים"]),
    ("הקוד הסודי שלי 9876 ואני רוצה תלונה על הפקח", ["9876"], ["תלונה", "פקח"]),
]
# Useful short numbers must be kept: last four digits are what the refund procedure asks for.
KEEP_CASE = ("אני רוצה החזר על חיוב כפול, 4 ספרות אחרונות 5566, נסעתי מאלנבי לארלוזורוב", "5566")


def unit_checks() -> list[str]:
    failures = []
    for text, secrets, keep in CASES:
        clean, kinds = redact_sensitive(text)
        for s in secrets:
            if s in clean:
                failures.append(f"unit: {s!r} survived in {clean!r}")
        if not kinds:
            failures.append(f"unit: nothing reported as redacted for {text!r}")
        for k in keep:
            if k not in clean:
                failures.append(f"unit: useful text {k!r} lost in {clean!r}")
        if MARKER not in clean:
            failures.append(f"unit: marker missing in {clean!r}")
    clean, kinds = redact_sensitive(KEEP_CASE[0])
    if KEEP_CASE[1] not in clean or kinds:
        failures.append(f"unit: last four digits must be kept, got {clean!r} kinds={kinds}")
    return failures


async def end_to_end_checks() -> list[str]:
    failures = []
    retriever = BM25Retriever(load_passages(config.DOCS_DIR))
    bridge = MCPBridge(config.MCP_SERVER_SCRIPT)
    await bridge.start()
    try:
        for text, secrets, keep in CASES:
            r = await run_free_question(text, [], retriever, bridge)
            if r["action"] != "handoff" or not r["case"]:
                failures.append(f"e2e: expected a handoff with a case for {text!r}, got action={r['action']}")
                continue
            case_id = r["case"]["case_id"]
            stored = next((c for c in json.loads(config.CASES_FILE.read_text(encoding="utf-8")) if c["case_id"] == case_id), None)
            if stored is None:
                failures.append(f"e2e: case {case_id} not found in {config.CASES_FILE.name}")
                continue
            stored_json = json.dumps(stored, ensure_ascii=False)
            call = next(c for c in r["tool_calls"] if c["tool"] == "prepare_support_case")
            call_json = json.dumps(call, ensure_ascii=False)     # input, output and log entry as the UI shows them
            for s in secrets:
                if s in stored_json:
                    failures.append(f"e2e: secret {s!r} stored in {case_id}")
                if s in call_json:
                    failures.append(f"e2e: secret {s!r} present in MCP call input/output/log for {case_id}")
                if s in r["answer_he"]:
                    failures.append(f"e2e: secret {s!r} shown in the reply for {case_id}")
            for k in keep:
                if k not in stored["summary_he"]:
                    failures.append(f"e2e: useful description {k!r} missing from stored summary of {case_id}")
            if "sensitive_details_redacted" not in r["flags"]:
                failures.append(f"e2e: redaction flag missing for {case_id}")
        # Last four digits survive end to end.
        r = await run_free_question(KEEP_CASE[0], [], retriever, bridge)
        if not r["case"] or KEEP_CASE[1] not in r["case"]["summary_he"]:
            failures.append("e2e: last four digits were removed from a legitimate refund request")
    finally:
        await bridge.stop()
    return failures


def main() -> None:
    failures = unit_checks() + asyncio.run(end_to_end_checks())
    if failures:
        print("FAILED:\n - " + "\n - ".join(failures))
        sys.exit(1)
    print(f"redaction test: {len(CASES)} sensitive questions and 1 keep case passed (unit + stored case + MCP log)")


if __name__ == "__main__":
    main()
