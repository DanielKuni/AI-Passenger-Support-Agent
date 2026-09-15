"""
The agent: retrieve -> call Claude with MCP tools -> validate -> return a structured result.

One hand-written tool loop (no framework). The result object is what the UI and the eval consume:
  action, answer_he, sources (validated), retrieved passages, tool calls, flags, usage.
"""
from __future__ import annotations

import json
import re

import anthropic

from . import config
from .mcp_client import MCPBridge
from .prompts import SYSTEM_PROMPT, build_user_message
from .rag import BM25Retriever

# Words that usually mean "right now" / "today" / a live condition. Cheap, transparent, and measured by the eval.
LIVE_STATUS_PATTERNS = [
    r"עכשיו", r"כרגע", r"היום", r"הבוקר", r"הערב", r"ברגע זה", r"נכון ל",
    r"עיכוב", r"עיכובים", r"תקלה", r"שיבוש", r"מושבת", r"סגור", r"סגורה", r"פועל", r"פועלת", r"עובד", r"עובדת",
    r"יש רכבות", r"אין רכבות", r"אוטובוס חלופי", r"אוטובוסים חלופיים",
]
LIVE_RE = re.compile("|".join(LIVE_STATUS_PATTERNS))


def needs_live_status(question: str) -> bool:
    return bool(LIVE_RE.search(question))


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


class Agent:
    def __init__(self, retriever: BM25Retriever, bridge: MCPBridge, client: anthropic.AsyncAnthropic | None = None):
        self.retriever = retriever
        self.bridge = bridge
        self.client = client or anthropic.AsyncAnthropic()

    async def _create(self, **kwargs):
        if config.ENABLE_REFUSAL_FALLBACK:
            return await self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        return await self.client.messages.create(**kwargs)

    async def answer(self, question: str, history: list[dict], scenario_label: str | None = None) -> dict:
        """history: previous turns as [{"role": "user"|"assistant", "content": str}, ...]."""
        # 1. Retrieval (always; cheap). For very short follow-ups, add the previous user turn to the query.
        query = question
        if len(question.split()) < 4:
            prev_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
            query = f"{prev_user} {question}"
        hits = self.retriever.retrieve(query, k=config.TOP_K)
        retrieved = [dict(p.as_dict(), score=round(s, 2)) for p, s in hits]
        live_hint = needs_live_status(question)

        # 2. Build messages: prior turns (plain text) + this turn with passages.
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": build_user_message(question, retrieved, live_hint, scenario_label)})

        tool_calls: list[dict] = []
        flags: list[str] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        final_text = ""
        log_start = len(self.bridge.call_log)

        # 3. Tool loop.
        for _round in range(config.MAX_TOOL_ROUNDS + 1):
            response = await self._create(
                model=config.MODEL_ID,
                max_tokens=config.MAX_TOKENS,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                tools=self.bridge.tools,
                messages=messages,
                output_config={"effort": config.EFFORT},
            )
            usage["input_tokens"] += response.usage.input_tokens
            usage["output_tokens"] += response.usage.output_tokens

            if response.stop_reason == "refusal":
                flags.append("model_refusal")
                final_text = json.dumps({"action": "unsupported",
                                         "answer_he": "לא הצלחתי לטפל בבקשה זו. מומלץ לפנות לנציג שירות.",
                                         "sources": [], "used_status_tool": False,
                                         "note": "stop_reason=refusal"}, ensure_ascii=False)
                break
            if response.stop_reason == "max_tokens":
                flags.append("max_tokens_hit")

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                final_text = "".join(b.text for b in response.content if b.type == "text")
                break

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:
                args = dict(tu.input) if isinstance(tu.input, dict) else json.loads(json.dumps(tu.input))
                text, is_error = await self.bridge.call_tool(tu.name, args)
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": text, "is_error": is_error})
            messages.append({"role": "user", "content": results})
        else:
            flags.append("tool_round_limit")

        tool_calls = self.bridge.call_log[log_start:]

        # 4. Parse + validate the structured answer.
        parsed = _extract_json(final_text)
        if parsed is None:
            flags.append("unparsed_model_output")
            parsed = {"action": "answer", "answer_he": final_text, "sources": [], "used_status_tool": False, "note": ""}
        action = parsed.get("action", "answer")
        if action not in ("answer", "clarify", "handoff", "unsupported"):
            flags.append(f"unknown_action:{action}")
            action = "answer"

        retrieved_ids = {p["id"] for p in retrieved}
        cited = [s for s in (parsed.get("sources") or []) if isinstance(s, str)]
        valid_sources = [s for s in cited if s in retrieved_ids]
        if len(valid_sources) != len(cited):
            flags.append("cited_unretrieved_passage")   # model cited an id that was never shown to it
        sources = [p for p in retrieved if p["id"] in valid_sources]

        status_called = any(c["tool"] == "get_service_status" for c in tool_calls)
        case_calls = [c for c in tool_calls if c["tool"] == "prepare_support_case" and not c["is_error"]]
        if live_hint and not status_called and action == "answer":
            flags.append("live_question_answered_without_status_tool")
        if action == "handoff" and not case_calls:
            flags.append("handoff_without_case")
        if action == "answer" and not sources and not status_called:
            flags.append("answer_without_sources_or_tools")

        case = None
        if case_calls:
            try:
                case = json.loads(case_calls[-1]["output"])
            except json.JSONDecodeError:
                case = {"raw": case_calls[-1]["output"]}

        return {
            "action": action,
            "answer_he": parsed.get("answer_he", ""),
            "note": parsed.get("note", ""),
            "sources": sources,
            "retrieved": retrieved,
            "live_hint": live_hint,
            "tool_calls": tool_calls,
            "case": case,
            "flags": flags,
            "usage": usage,
            "model": config.MODEL_ID,
            "raw_model_output": final_text,
        }
