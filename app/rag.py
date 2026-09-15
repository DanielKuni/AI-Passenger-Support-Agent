"""
Minimal, inspectable RAG layer.

- Documents: Markdown files in docs/ with a small front-matter block (id, title, version, status).
- Chunking: every "## " heading becomes one passage. Passage id = "<doc_id>#<n>".
- Retrieval: BM25 (implemented here, ~40 lines) over a Hebrew-aware tokenizer:
    * final letters normalised (ך ם ן ף ץ -> כ מ נ פ צ)
    * common one/two-letter prefixes (ו ה ב ל מ ש כ, וה, וב, ...) are stripped into extra tokens,
      so "בתחנה", "התחנה", "לתחנה" and "תחנה" all match.
No embeddings, no external index: the whole ranking can be explained on a whiteboard.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

FINAL_MAP = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
PREFIXES_2 = ("וה", "וב", "ול", "ומ", "וש", "וכ", "כש", "שה", "שב", "של", "מה", "לכ", "בה")
PREFIXES_1 = ("ו", "ה", "ב", "ל", "מ", "ש", "כ")
TOKEN_RE = re.compile(r"[א-תA-Za-z0-9]+")

# Tiny query-side expansion for passenger phrasing that the documents express differently.
# Applied to queries only, so the index stays exactly the document text.
QUERY_SYNONYMS = {
    "חויבתי": ["חיוב"], "חייבו": ["חיוב"], "חיובים": ["חיוב"],
    "פעמיים": ["כפול"], "שולם": ["תשלום"], "לשלם": ["תשלום"],
    "מעליות": ["מעלית"], "קנסו": ["קנס"], "נקנסתי": ["קנס"],
    "איבדתי": ["אבד", "אבדות"], "שכחתי": ["אבדות", "נשכח"],
    "החזר": ["החזרים"], "עגלה": ["עגלת"],
    # common plurals -> the singular form used in the documents
    "רכבות": ["רכבת"], "אוטובוסים": ["אוטובוס"], "תחנות": ["תחנה"], "שיבושים": ["שיבוש"], "עיכובים": ["עיכוב"],
}


def tokenize(text: str, expand_query: bool = False) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        t = raw.translate(FINAL_MAP)
        tokens.append(t)
        if expand_query:
            tokens.extend(QUERY_SYNONYMS.get(t, []))
        if len(t) >= 4:
            for p in PREFIXES_2:
                if t.startswith(p) and len(t) - 2 >= 2:
                    tokens.append(t[2:])
                    break
            for p in PREFIXES_1:
                if t.startswith(p) and len(t) - 1 >= 2:
                    tokens.append(t[1:])
                    break
    return tokens


@dataclass
class Passage:
    id: str
    doc_id: str
    doc_title: str
    doc_status: str
    doc_version: str
    section_title: str
    text: str
    tokens: list[str] = field(default_factory=list, repr=False)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "doc_status": self.doc_status,
            "doc_version": self.doc_version,
            "section_title": self.section_title,
            "text": self.text,
        }


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    meta: dict[str, str] = {}
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            for line in raw[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            raw = raw[end + 4:]
    return meta, raw


def load_passages(docs_dir: Path) -> list[Passage]:
    passages: list[Passage] = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        doc_id = meta.get("id", path.stem)
        sections = re.split(r"^## ", body, flags=re.M)
        n = 0
        for sec in sections:
            sec = sec.strip()
            if not sec or "\n" not in sec:
                continue
            title, text = sec.split("\n", 1)
            n += 1
            p = Passage(
                id=f"{doc_id}#{n}",
                doc_id=doc_id,
                doc_title=meta.get("title", path.stem),
                doc_status=meta.get("status", ""),
                doc_version=meta.get("version", ""),
                section_title=title.strip(),
                text=text.strip(),
            )
            # Index title + doc title + text so a query about the topic hits the right doc.
            p.tokens = tokenize(f"{p.doc_title} {p.section_title} {p.text}")
            passages.append(p)
    return passages


class BM25Retriever:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.N = len(passages)
        self.avgdl = sum(len(p.tokens) for p in passages) / max(self.N, 1)
        self.tf: list[Counter] = [Counter(p.tokens) for p in passages]
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        # BM25 idf with the usual +1 smoothing so rare terms weigh more.
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, query_tokens: list[str], i: int) -> float:
        tf, dl = self.tf[i], len(self.passages[i].tokens)
        s = 0.0
        for t in set(query_tokens):
            if t not in tf:
                continue
            f = tf[t]
            s += self.idf[t] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def retrieve(self, query: str, k: int = 4) -> list[tuple[Passage, float]]:
        q = tokenize(query, expand_query=True)
        scored = [(self.passages[i], self.score(q, i)) for i in range(self.N)]
        scored = [(p, s) for p, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def get(self, passage_id: str) -> Passage | None:
        return next((p for p in self.passages if p.id == passage_id), None)


if __name__ == "__main__":  # quick manual check:  python -m app.rag "חיוב כפול"
    import sys
    from .config import DOCS_DIR

    r = BM25Retriever(load_passages(DOCS_DIR))
    print(f"{r.N} passages")
    for p, s in r.retrieve(" ".join(sys.argv[1:]) or "חיוב כפול בכרטיס אשראי"):
        print(f"{s:6.2f}  {p.id:22s} {p.doc_title} / {p.section_title}")
