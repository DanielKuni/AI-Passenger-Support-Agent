"""Central configuration. Everything that an interviewer might ask "why this value?" lives here."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DOCS_DIR = ROOT / "docs"
MCP_SERVER_SCRIPT = ROOT / "mcp_server" / "server.py"
STATUS_FILE = ROOT / "mcp_server" / "demo_status.json"
CASES_FILE = ROOT / "mcp_server" / "demo_cases.json"

# Model choice: Claude Opus 5 is the current default recommendation for tool-using agents.
MODEL_ID = os.getenv("MODEL_ID", "claude-opus-5")
# Effort: "medium" keeps latency and cost reasonable for a support chat; raise to "high" if needed.
EFFORT = os.getenv("EFFORT", "medium")
MAX_TOKENS = 4096          # answers are short Hebrew paragraphs; this is plenty
MAX_TOOL_ROUNDS = 5        # hard cap on the tool-use loop
TOP_K = 4                  # passages injected into the prompt
ENABLE_REFUSAL_FALLBACK = os.getenv("ENABLE_REFUSAL_FALLBACK", "false").lower() == "true"


def has_api_key() -> bool:
    """True only for a real-looking credential. The placeholder from .env.example ("sk-ant-...") does not count,
    so copying the example file verbatim keeps the app in offline demo mode."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        v = (os.getenv(name) or "").strip()
        if v and not v.endswith("...") and len(v) >= 20:
            return True
    return False
