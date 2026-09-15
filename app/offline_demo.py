"""
Offline demo mode: three guided passenger scenarios that run WITHOUT any model or API key.

What is real in this mode:
  - retrieval: the passenger question goes through the real BM25 retriever over docs/,
  - MCP: the listed tool calls are sent to the real MCP server and their real results are logged,
  - validation: predefined source ids are checked against what retrieval actually returned.

What is NOT real: the reply text. It is a PREDEFINED DEMO RESPONSE written by hand for each scenario,
optionally filled with values taken from the real tool result (status summary, case id). No language
model is called and nothing here pretends to be one. The UI labels every such reply accordingly.

Run from a terminal:  python -m app.offline_demo
"""
from __future__ import annotations

import asyncio
import json
import sys

from . import config
from .mcp_client import MCPBridge
from .rag import BM25Retriever, load_passages

PREDEFINED_LABEL_HE = "תשובה מוגדרת מראש במצב הדגמה ללא מודל. היא לא נוצרה על ידי מודל שפה."

SCENARIOS: list[dict] = [
    {
        "id": "payment_rag",
        "title_he": "1. שאלת תשלום: אחזור ממסמכים (RAG)",
        "intro_he": "השאלה עוברת אחזור אמיתי (BM25) במסמכי ההדגמה. לא נקרא אף כלי MCP. התשובה מוגדרת מראש ומצטטת רק פסקאות שאוחזרו בפועל.",
        "question": "אפשר לשלם במזומן ברכבת?",
        "status_scenario": "normal",
        "tool_calls": [],
        "action": "answer",
        "sources": ["payment_methods#1"],
        "response_template_he": (
            "לא. לפי מסמך ההדגמה \"אמצעי תשלום ותיקוף\", לא ניתן לשלם במזומן בתחנות או ברכבת. "
            "אפשר לשלם בכרטיס רב-קו טעון, בכרטיס אשראי או חיוב עם תשלום ללא מגע (הצמדה לקורא בכניסה וביציאה), "
            "או באפליקציית תשלום מאושרת."
        ),
    },
    {
        "id": "status_mcp",
        "title_he": "2. שאלת מצב שירות: קריאה אמיתית לכלי MCP",
        "intro_he": "התרחיש מעביר את מערכת מצב השירות (הדגמה) למצב \"קטע סגור\", מריץ אחזור, ואז קורא בפועל לכלי get_service_status בשרת ה-MCP. התשובה היא תבנית מוגדרת מראש שממולאת בערכים שהכלי החזיר.",
        "question": "אני בארלוזורוב ורוצה להגיע לאלנבי. יש רכבות עכשיו או שיש שיבוש בקטע?",
        "status_scenario": "segment_closed",
        "tool_calls": [{"tool": "get_service_status", "args": {"station": "ארלוזורוב"}}],
        "action": "answer",
        "sources": ["disruption_policy#2"],
        "response_template_he": (
            "לפי מערכת מצב השירות (נתוני הדגמה, נבדקו כעת): {status_summary} {status_alternative} "
            "לפי מסמך המדיניות (הדגמה), הנסיעה באוטובוס החלופי אינה כרוכה בתשלום נוסף למי שכבר תיקף בקו האדום."
        ),
    },
    {
        "id": "handoff_mcp",
        "title_he": "3. העברה לנציג: הכנת פניית הדגמה דרך MCP",
        "intro_he": "האחזור מוצא את מדיניות ההסלמה (ערעור על קנס מטופל על ידי נציג אנושי). הקוד קורא בפועל לכלי prepare_support_case בשרת ה-MCP, שמייצר פניית הדגמה ומחזיר מספר פנייה. הפנייה נשמרת במערכת ההדגמה בלבד ואינה נשלחת לצוות שירות אמיתי. התשובה מוגדרת מראש ומשלבת את מספר הפנייה שהכלי החזיר.",
        "question": "קיבלתי קנס למרות שתיקפתי, אני רוצה לערער",
        "status_scenario": "normal",
        "tool_calls": [{
            "tool": "prepare_support_case",
            "args": {
                "category": "fine_appeal",
                "summary_he": "נוסע מבקש לערער על קנס פיקוח; לדבריו תיקף לפני הנסיעה.",
                "passenger_details_he": "הנוסע מסר: קיבל קנס למרות שתיקף. לא נמסרו תחנה, שעה או אמצעי תשלום.",
                "checks_done_he": "נמצאה הנחיית הסלמה במסמכי ההדגמה: ערעור על קנס מטופל על ידי נציג אנושי בלבד.",
            },
        }],
        "action": "handoff",
        "sources": ["escalation_policy#1", "escalation_policy#3"],
        "response_template_he": (
            "ערעור על קנס מטופל על ידי נציג אנושי בלבד, ולכן הוכנה פניית הדגמה מספר {case_id}. "
            "הפנייה נשמרת במערכת ההדגמה בלבד ואינה נשלחת לצוות שירות אמיתי. "
            "בפנייה אמיתית כדאי לצרף תחנה, שעה ואמצעי התשלום שבו תיקפת. אני לא יכול לקבוע אם הקנס יבוטל."
        ),
    },
]


def list_scenarios() -> list[dict]:
    return [{k: s[k] for k in ("id", "title_he", "intro_he", "question", "status_scenario")} for s in SCENARIOS]


def _set_status_scenario(name: str) -> None:
    data = json.loads(config.STATUS_FILE.read_text(encoding="utf-8"))
    data["active_scenario"] = name
    config.STATUS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_scenario(scenario_id: str, retriever: BM25Retriever, bridge: MCPBridge) -> dict:
    s = next((x for x in SCENARIOS if x["id"] == scenario_id), None)
    if s is None:
        raise KeyError(scenario_id)
    steps: list[dict] = []
    flags: list[str] = []

    # 0. Put the simulated status feed in the state the scenario needs (and say so).
    _set_status_scenario(s["status_scenario"])
    steps.append({"step": "status_scenario", "detail_he": f"מערכת מצב השירות (הדגמה) הועברה לתרחיש \"{s['status_scenario']}\"."})

    # 1. Real retrieval.
    hits = retriever.retrieve(s["question"], k=config.TOP_K)
    retrieved = [dict(p.as_dict(), score=round(sc, 2)) for p, sc in hits]
    steps.append({"step": "retrieval", "detail_he": f"אחזור BM25 אמיתי: {len(retrieved)} פסקאות אוחזרו מתוך {retriever.N}."})

    # 2. Real MCP calls.
    log_start = len(bridge.call_log)
    fill: dict[str, str] = {"status_summary": "", "status_alternative": "", "case_id": ""}
    tool_error: str | None = None
    for call in s["tool_calls"]:
        text, is_error = await bridge.call_tool(call["tool"], call["args"])
        steps.append({"step": "mcp_call", "detail_he": f"קריאה אמיתית לכלי MCP {call['tool']}: {'שגיאה' if is_error else 'הצליחה'}."})
        if is_error:
            tool_error = f"{call['tool']}: {text}"
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {}
        if call["tool"] == "get_service_status":
            fill["status_summary"] = payload.get("summary_he", "")
            dis = payload.get("disruptions") or []
            fill["status_alternative"] = dis[0].get("alternative_he", "") if dis else ""
        elif call["tool"] == "prepare_support_case":
            fill["case_id"] = payload.get("case_id", "")
    tool_calls = bridge.call_log[log_start:]

    # 3. Validate predefined sources against what was really retrieved (same rule as the live agent).
    retrieved_ids = {p["id"] for p in retrieved}
    valid = [sid for sid in s["sources"] if sid in retrieved_ids]
    if len(valid) != len(s["sources"]):
        flags.append("predefined_source_not_retrieved")
    sources = [p for p in retrieved if p["id"] in valid]

    # 4. Predefined response (never a model).
    if tool_error:
        answer = f"כלי ה-MCP החזיר שגיאה, ולכן אין תשובה מוגדרת מראש לתרחיש זה: {tool_error}"
        action = "unsupported"
        flags.append("mcp_tool_error")
    else:
        answer = s["response_template_he"].format(**fill).replace("  ", " ").strip()
        action = s["action"]
    steps.append({"step": "predefined_response", "detail_he": PREDEFINED_LABEL_HE})

    case = None
    case_calls = [c for c in tool_calls if c["tool"] == "prepare_support_case" and not c["is_error"]]
    if case_calls:
        try:
            case = json.loads(case_calls[-1]["output"])
        except json.JSONDecodeError:
            case = {"raw": case_calls[-1]["output"]}

    return {
        "mode": "offline_demo",
        "response_kind": "predefined",
        "response_label_he": PREDEFINED_LABEL_HE,
        "scenario_id": s["id"],
        "scenario_title_he": s["title_he"],
        "question": s["question"],
        "status_scenario": s["status_scenario"],
        "action": action,
        "answer_he": answer,
        "note": "predefined demo response; no model call",
        "sources": sources,
        "retrieved": retrieved,
        "live_hint": None,
        "tool_calls": tool_calls,
        "case": case,
        "flags": flags,
        "steps": steps,
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "model": None,
        "raw_model_output": "",
        "demo": True,
    }


async def _cli() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    retriever = BM25Retriever(load_passages(config.DOCS_DIR))
    bridge = MCPBridge(config.MCP_SERVER_SCRIPT)
    await bridge.start()
    problems = 0
    try:
        for s in SCENARIOS:
            print("\n" + "=" * 78)
            print(s["title_he"])
            print("שאלת הנוסע:", s["question"])
            r = await run_scenario(s["id"], retriever, bridge)
            print("\nפסקאות שאוחזרו (BM25):")
            for p in r["retrieved"]:
                mark = "*" if any(src["id"] == p["id"] for src in r["sources"]) else " "
                print(f"  {mark} {p['score']:6.2f}  {p['id']:22s} {p['doc_title']} / {p['section_title']}")
            print("\nקריאות MCP אמיתיות:")
            if not r["tool_calls"]:
                print("  (אין)")
            for c in r["tool_calls"]:
                print(f"  {c['tool']}({json.dumps(c['input'], ensure_ascii=False)}) -> {'ERROR ' if c['is_error'] else ''}{c['output'][:200].replace(chr(10), ' ')}")
            print(f"\n[{r['response_label_he']}]")
            print(f"פעולה: {r['action']}")
            print(r["answer_he"])
            if r["flags"]:
                problems += 1
                print("דגלים:", r["flags"])
    finally:
        _set_status_scenario("normal")
        await bridge.stop()
    print("\n" + ("offline demo: all scenarios ran without flags" if not problems else f"offline demo: {problems} scenario(s) raised flags"))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_cli()))
