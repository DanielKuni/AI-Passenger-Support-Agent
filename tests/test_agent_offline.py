"""
Offline test of the agent loop with a STUB model (no API key, no network).

It checks the plumbing an interviewer will ask about:
  - passages are retrieved and injected,
  - a tool_use block from the model is routed to the real MCP server and the result is fed back,
  - the final JSON is parsed, cited sources are validated against what was retrieved,
  - code-side flags fire (e.g. a live question answered without the status tool).

Run:  python -m tests.test_agent_offline
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app import config  # noqa: E402
from app.agent import Agent, needs_live_status  # noqa: E402
from app.mcp_client import MCPBridge  # noqa: E402
from app.rag import BM25Retriever, load_passages  # noqa: E402


class StubMessages:
    """Scripted responses: first a tool call, then a final JSON answer."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.script.pop(0)


def text_block(t):
    return SimpleNamespace(type="text", text=t)


def tool_block(name, inp, id_="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=id_)


def response(content, stop_reason):
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=10, output_tokens=5))


async def main():
    retriever = BM25Retriever(load_passages(config.DOCS_DIR))
    bridge = MCPBridge(config.MCP_SERVER_SCRIPT)
    await bridge.start()
    failures = []
    try:
        # --- Case 1: live question -> model calls status tool -> final JSON citing a real passage id.
        q = "יש עיכובים בקו עכשיו?"
        assert needs_live_status(q)
        hits = retriever.retrieve(q, k=config.TOP_K)
        some_id = hits[0][0].id if hits else "disruption_policy#1"
        stub = StubMessages([
            response([tool_block("get_service_status", {"station": None})], "tool_use"),
            response([text_block(json.dumps({"action": "answer", "answer_he": "הקו פועל כסדרו (הדגמה).",
                                             "sources": [some_id, "made_up#9"], "used_status_tool": True,
                                             "note": "stub"}, ensure_ascii=False))], "end_turn"),
        ])
        agent = Agent(retriever, bridge, client=SimpleNamespace(messages=stub))
        r = await agent.answer(q, [], scenario_label="normal")
        if [t["tool"] for t in r["tool_calls"]] != ["get_service_status"]:
            failures.append(f"case1: expected one status call, got {r['tool_calls']}")
        if r["action"] != "answer" or "cited_unretrieved_passage" not in r["flags"]:
            failures.append(f"case1: expected flag for made-up citation, got {r['flags']}")
        if [s["id"] for s in r["sources"]] != ([some_id] if hits else []):
            failures.append(f"case1: sources not validated: {[s['id'] for s in r['sources']]}")
        # The second request must carry the tool_result back to the model.
        second = stub.requests[1]["messages"]
        if not (second[-1]["role"] == "user" and second[-1]["content"][0]["type"] == "tool_result"):
            failures.append("case1: tool_result was not appended to messages")
        first_user = stub.requests[0]["messages"][0]["content"]  # index 0: the list is mutated by later rounds
        if "<passages>" not in first_user or "Code-side classifier" not in first_user:
            failures.append("case1: passages/hint missing from user message")

        # --- Case 2: live question answered WITHOUT the tool -> flag must fire.
        stub2 = StubMessages([response([text_block(json.dumps({"action": "answer", "answer_he": "אין עיכובים.", "sources": []}))], "end_turn")])
        r2 = await Agent(retriever, bridge, client=SimpleNamespace(messages=stub2)).answer(q, [], "normal")
        if "live_question_answered_without_status_tool" not in r2["flags"]:
            failures.append(f"case2: missing flag, got {r2['flags']}")

        # --- Case 3: handoff without a case -> flag; handoff with case -> case parsed.
        stub3 = StubMessages([
            response([tool_block("prepare_support_case", {"category": "fine_appeal", "summary_he": "ערעור על קנס (בדיקה)"})], "tool_use"),
            response([text_block(json.dumps({"action": "handoff", "answer_he": "הכנתי פנייה.", "sources": []}))], "end_turn"),
        ])
        r3 = await Agent(retriever, bridge, client=SimpleNamespace(messages=stub3)).answer("קיבלתי קנס ואני רוצה לערער", [], "normal")
        if not (r3["case"] and r3["case"].get("case_id", "").startswith("DEMO-")):
            failures.append(f"case3: case not parsed: {r3['case']}")
        if "handoff_without_case" in r3["flags"]:
            failures.append("case3: wrong flag")

        # --- Case 4: unparseable model output degrades gracefully.
        stub4 = StubMessages([response([text_block("סתם טקסט בלי JSON")], "end_turn")])
        r4 = await Agent(retriever, bridge, client=SimpleNamespace(messages=stub4)).answer("שלום", [], "normal")
        if "unparsed_model_output" not in r4["flags"] or r4["answer_he"] != "סתם טקסט בלי JSON":
            failures.append(f"case4: bad fallback {r4['flags']} {r4['answer_he']!r}")

        # --- Case 5: status feed outage -> tool result is an error and reaches the model as is_error.
        from eval.run_eval import set_scenario
        set_scenario("status_unavailable")
        stub5 = StubMessages([
            response([tool_block("get_service_status", {})], "tool_use"),
            response([text_block(json.dumps({"action": "unsupported", "answer_he": "לא הצלחתי לוודא.", "sources": []}))], "end_turn"),
        ])
        r5 = await Agent(retriever, bridge, client=SimpleNamespace(messages=stub5)).answer("הרכבת פועלת היום?", [], "status_unavailable")
        set_scenario("normal")
        if not (r5["tool_calls"] and r5["tool_calls"][0]["is_error"]):
            failures.append(f"case5: expected error tool result, got {r5['tool_calls']}")
        if not stub5.requests[1]["messages"][-1]["content"][0].get("is_error"):
            failures.append("case5: is_error not propagated to the model")
    finally:
        await bridge.stop()

    if failures:
        print("FAILED:\n - " + "\n - ".join(failures))
        sys.exit(1)
    print("offline agent test: all 5 cases passed")


if __name__ == "__main__":
    asyncio.run(main())
