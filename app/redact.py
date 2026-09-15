"""
Redaction of sensitive details in passenger text before it reaches a demo case, an MCP call or a log.

What is removed (values are replaced by a marker, never echoed anywhere):
  - card-like numbers: 13 to 19 digits, with or without spaces or hyphens
  - identity-like numbers: runs of 9 or more digits
  - secrets that follow a keyword such as סיסמה, קוד סודי, PIN, password, CVV, תעודת זהות:
    the first digit-bearing token or Latin word after the keyword, within the same sentence
What is kept on purpose: short numbers such as "4 ספרות אחרונות 5566", dates, times, bus numbers and station names,
because the guidance documents ask for exactly those to investigate a charge.

The same function is applied in three places: the offline workflow (before the case is composed), the MCP client
(before the input is sent and logged) and the MCP server (before the case is written to disk).
"""
from __future__ import annotations

import re

MARKER = "[הוסר]"

CARD_RE = re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)")
LONG_ID_RE = re.compile(r"(?<!\d)\d{9,}(?!\d)")
SECRET_KEYWORD_RE = re.compile(
    r"(?:ה?סיסמ[הא]|סיסמתי|ה?קוד\s+ה?(?:סודי|אישי|גישה|אימות)|pin|password|passcode|cvv|cvc|"
    r"תעודת\s+ה?זהות|ת\"ז|ת\.ז|מספר\s+ה?זהות|מספר\s+ת\"ז)",
    re.IGNORECASE,
)
# A secret value: any token containing a digit, or a Latin word of 4+ characters (passwords in a Hebrew sentence).
SECRET_VALUE_RE = re.compile(r"[^\s,.;:!?]*\d[^\s,.;:!?]*|[A-Za-z][A-Za-z0-9!@#$%^&*._\-]{3,}")
SENTENCE_END_RE = re.compile(r"[.;!?\n]")


def redact_sensitive(text: str) -> tuple[str, list[str]]:
    """Return (clean_text, kinds). kinds lists what was removed, e.g. ["card_number", "secret_after_keyword"]."""
    kinds: list[str] = []
    if not text:
        return text, kinds

    def _card(m: re.Match) -> str:
        kinds.append("card_number")
        return MARKER

    text = CARD_RE.sub(_card, text)

    def _long_id(m: re.Match) -> str:
        kinds.append("long_number")
        return MARKER

    text = LONG_ID_RE.sub(_long_id, text)

    # Keyword-triggered secrets: look at the rest of the sentence after each keyword.
    out, pos = [], 0
    for kw in SECRET_KEYWORD_RE.finditer(text):
        if kw.start() < pos:
            continue
        end = SENTENCE_END_RE.search(text, kw.end())
        window_end = end.start() if end else len(text)
        window = text[kw.end():window_end]
        val = SECRET_VALUE_RE.search(window)
        if val and val.group(0) != MARKER:
            out.append(text[pos:kw.end() + val.start()])
            out.append(MARKER)
            pos = kw.end() + val.end()
            kinds.append("secret_after_keyword")
    out.append(text[pos:])
    return "".join(out), kinds


def redact_mapping(values: dict) -> tuple[dict, list[str]]:
    """Apply redact_sensitive to every string value of a flat dict (tool arguments, case fields)."""
    clean, kinds = {}, []
    for k, v in values.items():
        if isinstance(v, str):
            v, k2 = redact_sensitive(v)
            kinds.extend(k2)
        clean[k] = v
    return clean, kinds
