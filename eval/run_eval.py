"""
Evaluation runner.

  python -m eval.run_eval            # both parts (agent part is skipped if no API key)
  python -m eval.run_eval --retrieval-only

Part A - retrieval check (no API key needed): for every case with `expected_doc`, is a passage from that
         document among the top-4 BM25 results?  (hit@4)
Part B - end-to-end agent check (needs ANTHROPIC_API_KEY): runs each case through the real agent + MCP server
         and applies the deterministic checks listed in cases.json. No LLM judge, no invented scores.

Writes eval/results.md (human readable) and eval/results_raw.json (full outputs).
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app import config  # noqa: E402
from app.rag import BM25Retriever, load_passages  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
ALL_CASES = [c for c in json.loads((EVAL_DIR / "cases.json").read_text(encoding="utf-8"))["cases"] if not c.get("skip")]
CASES = [c for c in ALL_CASES if not c.get("paraphrase") and not c.get("blind")]   # the original 24 (used while tuning)
PARAPHRASE_CASES = [c for c in ALL_CASES if c.get("paraphrase")]   # 10 additional paraphrases; consulted during debugging, not blind
BLIND_CASES = [c for c in ALL_CASES if c.get("blind")]             # written after the rules were frozen; run once, never tuned on


def set_scenario(name: str) -> None:
    data = json.loads(config.STATUS_FILE.read_text(encoding="utf-8"))
    data["active_scenario"] = name
    config.STATUS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def retrieval_eval(retriever: BM25Retriever) -> list[dict]:
    rows = []
    for c in CASES:
        if "expected_doc" not in c:
            continue
        query = c["turns"][-1] if len(c["turns"][-1].split()) >= 4 else " ".join(c["turns"])
        hits = retriever.retrieve(query, k=config.TOP_K)
        docs = [p.doc_id for p, _ in hits]
        rows.append({"id": c["id"], "expected_doc": c["expected_doc"], "top_docs": docs,
                     "hit": c["expected_doc"] in docs, "rank": docs.index(c["expected_doc"]) + 1 if c["expected_doc"] in docs else None})
    return rows


SENTENCE_SPLIT_RE = re.compile(r"(?<=[^0-9][.!?])\s+")   # same rule as app/offline_workflow.py


def spoken_summary_failures(r: dict) -> list[str]:
    """Global rule for every reply: a spoken summary exists, is short, and every sentence of it appears verbatim
    in the on-screen reply, so speech can never claim more than the text does."""
    failed = []
    spoken = " ".join((r.get("spoken_summary_he") or "").split())
    answer = " ".join((r.get("answer_he") or "").split())
    if not spoken:
        return ["spoken_summary missing"]
    sentences = [s for s in SENTENCE_SPLIT_RE.split(spoken) if s]
    if len(sentences) > 3:
        failed.append(f"spoken_summary has {len(sentences)} sentences (max 3)")
    if len(spoken) > 400:
        failed.append(f"spoken_summary too long ({len(spoken)} chars)")
    for s in sentences:
        if s not in answer:
            failed.append("spoken_summary sentence not verbatim in reply")
            break
    return failed


def check_case(c: dict, r: dict) -> list[str]:
    """Return a list of failed check names (empty = pass)."""
    failed = spoken_summary_failures(r)
    ans = r["answer_he"] or ""
    called = {t["tool"] for t in r["tool_calls"] if not t["is_error"]} | {t["tool"] for t in r["tool_calls"]}
    cited_docs = {s["doc_id"] for s in r["sources"]}
    if r["action"] not in c.get("expected_actions", [r["action"]]):
        failed.append(f"action={r['action']} expected {c['expected_actions']}")
    for t in c.get("must_call", []):
        if t not in called:
            failed.append(f"must_call {t}")
    for t in c.get("must_not_call", []):
        if t in called:
            failed.append(f"must_not_call {t}")
    if c.get("must_cite_any") and not (set(c["must_cite_any"]) & cited_docs):
        failed.append(f"must_cite_any {c['must_cite_any']} (cited {sorted(cited_docs)})")
    if c.get("answer_must_contain_any") and not any(s in ans for s in c["answer_must_contain_any"]):
        failed.append("answer_must_contain_any")
    for s in c.get("answer_must_contain_all", []):
        if s not in ans:
            failed.append(f"answer_must_contain '{s}'")
    for s in c.get("answer_must_not_contain", []):
        if s in ans:
            failed.append(f"answer_must_not_contain '{s}'")
    if c.get("answer_must_not_match") and re.search(c["answer_must_not_match"], ans):
        failed.append(f"answer_must_not_match /{c['answer_must_not_match']}/")
    for f in c.get("flags_must_not_include", []):
        if f in r["flags"]:
            failed.append(f"flag {f}")
    # Privacy: inspect what was actually sent to and stored by the MCP server, not only the visible reply.
    if c.get("case_must_not_contain"):
        stored = json.dumps(r.get("case") or {}, ensure_ascii=False)
        calls = json.dumps([t for t in r["tool_calls"] if t["tool"] == "prepare_support_case"], ensure_ascii=False)
        for s in c["case_must_not_contain"]:
            if s in stored or s in calls:
                failed.append(f"case_must_not_contain '{s}' (found in stored case or MCP call log)")
    if c.get("case_must_contain_any"):
        stored = json.dumps(r.get("case") or {}, ensure_ascii=False)
        if not any(s in stored for s in c["case_must_contain_any"]):
            failed.append("case_must_contain_any (stored case lacks the useful description)")
    return failed


async def agent_eval(retriever: BM25Retriever) -> list[dict]:
    import anthropic
    from app.agent import Agent
    from app.mcp_client import MCPBridge

    bridge = MCPBridge(config.MCP_SERVER_SCRIPT)
    await bridge.start()
    agent = Agent(retriever, bridge, anthropic.AsyncAnthropic())
    rows = []
    try:
        for c in CASES:
            set_scenario(c.get("scenario", "normal"))
            history: list[dict] = []
            result = None
            error = None
            try:
                for turn in c["turns"]:
                    result = await agent.answer(turn, history, scenario_label=c.get("scenario"))
                    history.append({"role": "user", "content": turn})
                    history.append({"role": "assistant", "content": result["answer_he"]})
            except Exception as e:  # keep going; record the failure
                error = f"{type(e).__name__}: {e}"
            if result is None:
                rows.append({"id": c["id"], "category": c["category"], "passed": False, "failed": [f"exception: {error}"],
                             "action": None, "tools": [], "answer_he": "", "flags": []})
                print(f"[{c['id']}] EXCEPTION {error}")
                continue
            failed = check_case(c, result)
            if error:
                failed.append(f"exception: {error}")
            rows.append({
                "id": c["id"], "category": c["category"], "passed": not failed, "failed": failed,
                "action": result["action"], "expected_actions": c.get("expected_actions"),
                "tools": [t["tool"] for t in result["tool_calls"]], "sources": [s["id"] for s in result["sources"]],
                "flags": result["flags"], "answer_he": result["answer_he"], "note": result.get("note", ""),
                "usage": result["usage"], "scenario": c.get("scenario", "normal"),
            })
            print(f"[{c['id']}] {'PASS' if not failed else 'FAIL ' + '; '.join(failed)} | action={result['action']} tools={rows[-1]['tools']}")
    finally:
        set_scenario("normal")
        await bridge.stop()
    return rows


async def offline_demo_eval(retriever: BM25Retriever) -> list[dict]:
    """Part C: the three guided offline scenarios. Real retrieval + real MCP; the reply text is predefined."""
    from app.mcp_client import MCPBridge
    from app.offline_demo import SCENARIOS, run_scenario

    bridge = MCPBridge(config.MCP_SERVER_SCRIPT)
    await bridge.start()
    rows = []
    try:
        for s in SCENARIOS:
            r = await run_scenario(s["id"], retriever, bridge)
            failed = []
            if not r["retrieved"]:
                failed.append("no passages retrieved")
            if [x["id"] for x in r["sources"]] != s["sources"]:
                failed.append(f"predefined sources not all retrieved: got {[x['id'] for x in r['sources']]}")
            expected_tools = [c["tool"] for c in s["tool_calls"]]
            if [c["tool"] for c in r["tool_calls"]] != expected_tools:
                failed.append(f"tool calls {[c['tool'] for c in r['tool_calls']]} != {expected_tools}")
            if any(c["is_error"] for c in r["tool_calls"]):
                failed.append("an MCP call returned an error")
            if s["action"] == "handoff" and not (r["case"] and str(r["case"].get("case_id", "")).startswith("DEMO-")):
                failed.append("no demo case id returned by MCP")
            if "{" in r["answer_he"]:
                failed.append("unfilled template placeholder")
            if r["flags"]:
                failed.append(f"flags {r['flags']}")
            failed += spoken_summary_failures(r)
            rows.append({"id": s["id"], "passed": not failed, "failed": failed, "action": r["action"],
                         "tools": [c["tool"] for c in r["tool_calls"]], "sources": [x["id"] for x in r["sources"]],
                         "case_id": (r["case"] or {}).get("case_id"), "answer_he": r["answer_he"]})
            print(f"[offline {s['id']}] {'PASS' if not failed else 'FAIL ' + '; '.join(failed)}")
    finally:
        set_scenario("normal")
        await bridge.stop()
    return rows


async def offline_workflow_eval(retriever: BM25Retriever, cases: list[dict]) -> list[dict]:
    """Parts D/E: cases through the deterministic offline workflow (free-form path), same checks as Part B."""
    from app.mcp_client import MCPBridge
    from app.offline_workflow import run_free_question

    bridge = MCPBridge(config.MCP_SERVER_SCRIPT)
    await bridge.start()
    rows = []
    try:
        for c in cases:
            set_scenario(c.get("scenario", "normal"))
            history: list[dict] = []
            result = None
            for turn in c["turns"]:
                result = await run_free_question(turn, history, retriever, bridge)
                history.append({"role": "user", "content": turn})
                history.append({"role": "assistant", "content": result["answer_he"]})
            failed = check_case(c, result)
            rows.append({"id": c["id"], "category": c["category"], "passed": not failed, "failed": failed,
                         "action": result["action"], "expected_actions": c.get("expected_actions"),
                         "tools": [t["tool"] for t in result["tool_calls"]], "sources": [s["id"] for s in result["sources"]],
                         "flags": result["flags"], "answer_he": result["answer_he"], "scenario": c.get("scenario", "normal")})
            print(f"[workflow {c['id']}] {'PASS' if not failed else 'FAIL ' + '; '.join(failed)} | action={result['action']} tools={rows[-1]['tools']}")
    finally:
        set_scenario("normal")
        await bridge.stop()
    return rows


def _workflow_section(title: str, rows: list[dict] | None, intro: str) -> list[str]:
    L = ["", f"## {title}", ""]
    if rows is None:
        return L + ["**Not run.**", ""]
    passed = sum(r["passed"] for r in rows)
    L += [f"**{passed}/{len(rows)} cases passed all their checks.** {intro}", ""]
    cats: dict[str, list] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r["passed"])
    L += ["| category | passed |", "|---|---|"]
    for k, v in cats.items():
        L.append(f"| {k} | {sum(v)}/{len(v)} |")
    L += ["", "| case | category | scenario | expected action | got | tools called | result |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        res = "PASS" if r["passed"] else "FAIL: " + "; ".join(r["failed"])
        L.append(f"| {r['id']} | {r['category']} | {r.get('scenario','')} | {', '.join(r.get('expected_actions') or [])} | {r['action']} | {', '.join(r['tools']) or '-'} | {res} |")
    L += ["", "### Templated replies (for manual reading)", ""]
    for r in rows:
        L += [f"**{r['id']}** ({r['action']}; sources {r.get('sources')})", "", f"> {r['answer_he'].replace(chr(10), ' ')}", ""]
    return L


def write_report(retrieval_rows: list[dict], agent_rows: list[dict] | None, skipped_reason: str | None,
                 offline_rows: list[dict] | None = None, workflow_rows: list[dict] | None = None,
                 paraphrase_rows: list[dict] | None = None, blind_rows: list[dict] | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    L = [f"# Evaluation results", "", f"Generated: {now}  ", f"Model: `{config.MODEL_ID}` (effort `{config.EFFORT}`)  ",
         f"Cases: {len(CASES)} in `eval/cases.json`", "",
         "All checks are deterministic string/tool/action checks defined per case. No LLM judge was used.", ""]
    hits = sum(r["hit"] for r in retrieval_rows)
    L += [f"## Part A - retrieval (BM25, hit@{config.TOP_K}) - no API key needed", "",
          f"**{hits}/{len(retrieval_rows)} cases** have a passage from the expected document in the top {config.TOP_K}.", "",
          "| case | expected doc | rank | top docs |", "|---|---|---|---|"]
    for r in retrieval_rows:
        L.append(f"| {r['id']} | {r['expected_doc']} | {r['rank'] if r['hit'] else 'miss'} | {', '.join(r['top_docs'])} |")
    L.append("")
    L.append("## Part B - end-to-end agent checks (real model + real MCP server)")
    L.append("")
    if agent_rows is None:
        L += [f"**Not run.** {skipped_reason}", ""]
    else:
        passed = sum(r["passed"] for r in agent_rows)
        L += [f"**{passed}/{len(agent_rows)} cases passed all their checks.**", ""]
        cats: dict[str, list] = {}
        for r in agent_rows:
            cats.setdefault(r["category"], []).append(r["passed"])
        L += ["| category | passed |", "|---|---|"]
        for k, v in cats.items():
            L.append(f"| {k} | {sum(v)}/{len(v)} |")
        tin = sum(r.get("usage", {}).get("input_tokens", 0) for r in agent_rows)
        tout = sum(r.get("usage", {}).get("output_tokens", 0) for r in agent_rows)
        L += ["", f"Tokens used by the run: {tin} input / {tout} output.", "",
              "| case | category | scenario | expected action | got | tools called | result |", "|---|---|---|---|---|---|---|"]
        for r in agent_rows:
            res = "PASS" if r["passed"] else "FAIL: " + "; ".join(r["failed"])
            L.append(f"| {r['id']} | {r['category']} | {r.get('scenario','')} | {', '.join(r.get('expected_actions') or [])} | {r['action']} | {', '.join(r['tools']) or '-'} | {res} |")
        L += ["", "### Answers (for manual reading)", ""]
        for r in agent_rows:
            L += [f"**{r['id']}** ({r['action']}; sources {r.get('sources')}; flags {r['flags']})", "", f"> {r['answer_he'].replace(chr(10), ' ')}", ""]
    L += ["", "## Part C - offline guided demo (real retrieval + real MCP, predefined responses) - no API key needed", ""]
    if offline_rows is None:
        L += ["**Not run.**", ""]
    else:
        L += [f"**{sum(r['passed'] for r in offline_rows)}/{len(offline_rows)} scenarios passed.** "
              "The reply text in these scenarios is predefined, so this part checks only the real parts: "
              "retrieval returned the passages the predefined reply cites, the MCP calls happened and succeeded, "
              "and the handoff scenario received a demo case id from the MCP server.", "",
              "| scenario | action | MCP tools called | sources validated | case id | result |", "|---|---|---|---|---|---|"]
        for r in offline_rows:
            L.append(f"| {r['id']} | {r['action']} | {', '.join(r['tools']) or '-'} | {', '.join(r['sources'])} | {r['case_id'] or '-'} | {'PASS' if r['passed'] else 'FAIL: ' + '; '.join(r['failed'])} |")
    L += _workflow_section(
        f"Part D - deterministic offline workflow on the original {len(CASES)} cases (free-form path, no model, no API key)",
        workflow_rows,
        "The checks were written for the model-driven agent; this part shows how far regex rules + BM25 quotes + real MCP calls "
        "get without a model. These cases were available while the rules and the quote threshold were being set, so this is not a blind test.")
    L += _workflow_section(
        f"Part E - {len(PARAPHRASE_CASES)} additional paraphrase cases (not a blind test)",
        paraphrase_rows,
        "These questions were written after the first version of the rules but before later rule changes, and their results were "
        "consulted while debugging. Their result when first run, before those later changes, was 7/10. Same workflow and checks as Part D.")
    L += _workflow_section(
        f"Part F - {len(BLIND_CASES)} blind questions (written after the rules were frozen, run once, never tuned on)",
        blind_rows,
        "Same workflow and checks as Part D. The rules were not changed after seeing these results.")
    (EVAL_DIR / "results.md").write_text("\n".join(L), encoding="utf-8")
    (EVAL_DIR / "results_raw.json").write_text(json.dumps({"retrieval": retrieval_rows, "agent": agent_rows, "offline_demo": offline_rows,
                                                            "offline_workflow": workflow_rows, "paraphrase": paraphrase_rows, "blind": blind_rows},
                                                           ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {EVAL_DIR / 'results.md'}")


def main() -> None:
    retriever = BM25Retriever(load_passages(config.DOCS_DIR))
    retrieval_rows = retrieval_eval(retriever)
    print(f"Part A retrieval: {sum(r['hit'] for r in retrieval_rows)}/{len(retrieval_rows)} hit@{config.TOP_K}")
    agent_rows, skipped, offline_rows, workflow_rows, paraphrase_rows, blind_rows = None, None, None, None, None, None
    if "--retrieval-only" not in sys.argv:
        offline_rows = asyncio.run(offline_demo_eval(retriever))
        workflow_rows = asyncio.run(offline_workflow_eval(retriever, CASES))
        paraphrase_rows = asyncio.run(offline_workflow_eval(retriever, PARAPHRASE_CASES))
        if BLIND_CASES and "--skip-blind" not in sys.argv:
            blind_rows = asyncio.run(offline_workflow_eval(retriever, BLIND_CASES))
    if "--retrieval-only" in sys.argv:
        skipped = "Skipped by --retrieval-only."
    elif not config.has_api_key():
        skipped = "ANTHROPIC_API_KEY is not set (this project is presented in offline demo mode). Add a key to .env and run `python -m eval.run_eval` to measure it."
    else:
        agent_rows = asyncio.run(agent_eval(retriever))
    write_report(retrieval_rows, agent_rows, skipped, offline_rows, workflow_rows, paraphrase_rows, blind_rows)


if __name__ == "__main__":
    main()
