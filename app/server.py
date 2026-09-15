"""FastAPI app: serves the Hebrew UI and a small JSON API around the agent."""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from . import config, tts
from .agent import Agent
from .mcp_client import MCPBridge
from .offline_demo import list_scenarios, run_scenario
from .offline_workflow import run_free_question
from .rag import BM25Retriever, load_passages

STATIC = Path(__file__).parent / "static"
state: dict = {"sessions": {}}


def read_status_file() -> dict:
    return json.loads(config.STATUS_FILE.read_text(encoding="utf-8"))


def set_scenario(name: str) -> dict:
    data = read_status_file()
    if name not in data["scenarios"]:
        raise KeyError(name)
    data["active_scenario"] = name
    config.STATUS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever = BM25Retriever(load_passages(config.DOCS_DIR))
    bridge = MCPBridge(config.MCP_SERVER_SCRIPT)
    await bridge.start()
    state["retriever"] = retriever
    state["bridge"] = bridge
    state["agent"] = Agent(retriever, bridge) if config.has_api_key() else None
    print(f"[app] {retriever.N} passages indexed from {config.DOCS_DIR}", flush=True)
    if state["agent"] is None:
        print("[app] WARNING: no ANTHROPIC_API_KEY - chat endpoint will return an error", flush=True)
    try:
        yield
    finally:
        await bridge.stop()


app = FastAPI(title="AI Passenger Support Agent (DEMO)", lifespan=lifespan)


class ChatIn(BaseModel):
    session_id: str | None = None
    message: str


class ScenarioIn(BaseModel):
    name: str


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
async def meta():
    data = read_status_file()
    return {
        # "offline_demo": no API key -> only the guided scenarios (predefined responses) are available.
        # "live": a key is present -> free chat through Claude is enabled as well.
        "mode": "live" if config.has_api_key() else "offline_demo",
        "model": config.MODEL_ID if config.has_api_key() else None,
        "api_key_present": config.has_api_key(),
        "offline_scenarios": list_scenarios(),
        "tts": tts.info(),
        "passages": state["retriever"].N,
        "docs": sorted({(p.doc_id, p.doc_title, p.doc_version) for p in state["retriever"].passages}),
        "tools": [t["name"] for t in state["bridge"].tools],
        "scenario": data["active_scenario"],
        "scenarios": {k: v["label_he"] for k, v in data["scenarios"].items()},
    }


@app.post("/api/scenario")
async def api_set_scenario(body: ScenarioIn):
    try:
        data = set_scenario(body.name)
    except KeyError:
        raise HTTPException(400, "unknown scenario")
    return {"scenario": data["active_scenario"], "demo": True}


@app.get("/api/cases")
async def cases():
    if not config.CASES_FILE.exists():
        return []
    return json.loads(config.CASES_FILE.read_text(encoding="utf-8"))


class OfflineIn(BaseModel):
    scenario_id: str


@app.get("/api/offline/scenarios")
async def offline_scenarios():
    return list_scenarios()


@app.post("/api/offline/run")
async def offline_run(body: OfflineIn):
    """Guided offline scenario: real retrieval + real MCP calls + a predefined (labelled) response. No model."""
    try:
        return JSONResponse(await run_scenario(body.scenario_id, state["retriever"], state["bridge"]))
    except KeyError:
        raise HTTPException(400, "unknown scenario")


@app.post("/api/offline/ask")
async def offline_ask(body: ChatIn):
    """Free-form question (typed or spoken) through the deterministic offline workflow. No model."""
    sid = body.session_id or str(uuid.uuid4())
    history = state["sessions"].setdefault("offline:" + sid, [])
    result = await run_free_question(body.message, history, state["retriever"], state["bridge"])
    history.append({"role": "user", "content": body.message})
    history.append({"role": "assistant", "content": result["answer_he"]})
    scenario = read_status_file()["active_scenario"]
    return JSONResponse({"session_id": sid, "scenario": scenario, "status_scenario": scenario, **result})


class TtsIn(BaseModel):
    text: str


@app.post("/api/tts")
async def api_tts(body: TtsIn):
    """Local Hebrew speech with Piper (optional). Returns a WAV. 503 when the voice is not installed."""
    if not tts.available():
        raise HTTPException(503, "Local Hebrew voice not installed. See README, section Voice interface.")
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    data = await asyncio.to_thread(tts.synthesize_wav, text)
    return Response(content=data, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.post("/api/reset")
async def reset(body: ChatIn):
    state["sessions"].pop(body.session_id, None)
    state["sessions"].pop("offline:" + str(body.session_id), None)
    return {"ok": True}


@app.post("/api/chat")
async def chat(body: ChatIn):
    agent: Agent | None = state.get("agent")
    if agent is None:
        raise HTTPException(503, "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    sid = body.session_id or str(uuid.uuid4())
    history = state["sessions"].setdefault(sid, [])
    scenario = read_status_file()["active_scenario"]
    try:
        result = await agent.answer(body.message, history, scenario_label=scenario)
    except anthropic.AuthenticationError:
        raise HTTPException(401, "Anthropic API key rejected.")
    except anthropic.RateLimitError:
        raise HTTPException(429, "Rate limited by the Anthropic API. Try again shortly.")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Anthropic API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Could not reach the Anthropic API.")
    # Keep plain-text turns only; the next turn rebuilds passages/tools from scratch.
    history.append({"role": "user", "content": body.message})
    history.append({"role": "assistant", "content": result["answer_he"] or "(empty)"})
    return JSONResponse({"session_id": sid, "scenario": scenario, "demo": True, **result})
