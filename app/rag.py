"""
Minimal, inspectable RAG layer.

- Documents: Markdown files in docs/ with a small front-matter block (id, title, version, status).
- Chunking: every "## " heading becomes one passage. Passage id = "<doc_id>#<n>".
- Retrieval: BM25 (implemented here, ~40 lines) over a Hebrew-aware tokenizer:
    * final letters normalised (ך ם ן ף ץ -> כ מ נ פ צ)
    * common one/two-letter prefixes (ו ה ב ל מ ש כ, וה, וב, ...) are stripped into extra tokens,
      so "בתחנה", "התחנה", "לתחנה" and "תחנה" all match.
- Field weighting: a term in the passage's own section title counts SECTION_TITLE_WEIGHT times, in the body once,
  and in the document title only DOC_TITLE_WEIGHT. Without this, a document title such as
  "כללי נסיעה: ילדים, אופניים, חיות ומטען" makes every section of that document match "אופניים" equally,
  and the section that is actually about bicycles loses to sections that share generic verbs.
  Document frequency (for idf) counts a term only where it appears in a section title or body.
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
SECTION_TITLE_WEIGHT = 2.0
DOC_TITLE_WEIGHT = 0.25

# Tiny query-side expansion for passenger phrasing that the documents express differently.
# Applied to queries only, so the index stays exactly the document text.
QUERY_SYNONYMS = {
    "חויבתי": ["חיוב"], "חייבו": ["חיוב"], "חיובים": ["חיוב"],
    "פעמיים": ["כפול"], "שולם": ["תשלום"], "לשלם": ["תשלום"],
    "קנסו": ["קנס"], "נקנסתי": ["קנס"],
    "איבדתי": ["אבד", "אבדות"], "שכחתי": ["אבדות", "נשכח"],
    "החזר": ["החזרים"], "עגלה": ["עגלת"],
    # passenger verbs -> the verbs the documents use for the same situation
    "נקלט": ["מגיב", "תיקוף"], "נקלטה": ["מגיב", "תיקוף"], "נקלטו": ["מגיב", "תיקוף"],
    "נתקעת": ["נעצרת"], "נתקעה": ["נעצרה"], "נתקע": ["נעצר"], "תקועה": ["נעצרה"], "תקוע": ["נעצר"],
}

# Words that carry no topic on their own. Excluded from "does this passage match the question" checks.
STOPWORDS = {w.translate(FINAL_MAP) for w in (
    "אם", "יש", "מה", "של", "את", "על", "לא", "או", "גם", "זה", "זו", "אני", "הוא", "היא", "עם", "כל", "איך",
    "האם", "לי", "שלי", "אפשר", "מותר", "צריך", "רוצה", "כמה", "מי", "איפה", "מתי", "למה", "בין", "אבל",
)}


def _suffix_variants(t: str) -> list[str]:
    """Light Hebrew suffix normalisation: plural and construct forms map to the base form used in the documents.
    שערים -> שער, מעליות -> מעלית, תחנות -> תחנה, תחנת -> תחנה, קטנות -> קטן."""
    out: list[str] = []
    if t.endswith("ימ") and len(t) >= 5:
        out.append(t[:-2])
    if t.endswith("יות") and len(t) >= 5:
        out.append(t[:-3] + "ית")
    elif t.endswith("ות") and len(t) >= 4:
        out.append(t[:-2] + "ה")
        out.append(t[:-2])
    elif t.endswith("ת") and len(t) >= 4:
        out.append(t[:-1] + "ה")
    return out


def tokenize_groups(text: str, expand_query: bool = False) -> list[list[str]]:
    """One group per word: the word itself plus its prefix-stripped / synonym variants.
    Scoring takes the best variant per group, so a word is never counted twice."""
    groups: list[list[str]] = []
    for raw in TOKEN_RE.findall(text.lower()):
        t = raw.translate(FINAL_MAP)
        variants = [t]
        if expand_query:
            variants.extend(QUERY_SYNONYMS.get(t, []))
        if len(t) >= 4:
            for p in PREFIXES_2:
                if t.startswith(p) and len(t) - 2 >= 2:
                    variants.append(t[2:])
                    break
            for p in PREFIXES_1:
                if t.startswith(p) and len(t) - 1 >= 2:
                    variants.append(t[1:])
                    break
        for v in list(variants):
            variants.extend(_suffix_variants(v))
        seen: set[str] = set()
        groups.append([v for v in variants if not (v in seen or seen.add(v))])
    return groups


def content_tokens(text: str, expand_query: bool = False) -> list[str]:
    """Query tokens without stopwords (used for match checks, not for scoring)."""
    return [v for g in tokenize_groups(text, expand_query) if g[0] not in STOPWORDS for v in g]


def tokenize(text: str, expand_query: bool = False) -> list[str]:
    return [v for g in tokenize_groups(text, expand_query) for v in g]


@dataclass
class Passage:
    id: str
    doc_id: str
    doc_title: str
    doc_status: str
    doc_version: str
    section_title: str
    text: str
    tokens: list[str] = field(default_factory=list, repr=False)          # section title + body tokens (length + df)
    weighted_tf: dict = field(default_factory=dict, repr=False)          # term -> weighted term frequency
    own_terms: set = field(default_factory=set, repr=False)              # terms in section title or body only
    section_terms: set = field(default_factory=set, repr=False)          # terms in the section title only

    def matches_in_own_text(self, query_tokens: list[str]) -> bool:
        """True if at least one query term occurs in this passage's own section title or body."""
        return any(t in self.own_terms for t in query_tokens)

    def matches_section_title(self, query_tokens: list[str]) -> bool:
        """True if a query term occurs in this passage's section title (the section is explicitly about it)."""
        return any(t in self.section_terms for t in query_tokens)

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
            # Field-weighted index: section title (x2), body (x1), document title (x0.25).
            section_tokens = tokenize(p.section_title)
            body_tokens = tokenize(p.text)
            doc_tokens = tokenize(p.doc_title)
            p.tokens = section_tokens + body_tokens
            p.own_terms = set(p.tokens)
            p.section_terms = set(section_tokens)
            wtf: dict[str, float] = {}
            for t in section_tokens:
                wtf[t] = wtf.get(t, 0.0) + SECTION_TITLE_WEIGHT
            for t in body_tokens:
                wtf[t] = wtf.get(t, 0.0) + 1.0
            for t in doc_tokens:
                wtf[t] = wtf.get(t, 0.0) + DOC_TITLE_WEIGHT
            p.weighted_tf = wtf
            passages.append(p)
    return passages


class BM25Retriever:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.N = len(passages)
        self.avgdl = sum(len(p.tokens) for p in passages) / max(self.N, 1)
        self.tf: list[dict] = [p.weighted_tf for p in passages]
        df: Counter = Counter()
        for p in passages:
            df.update(p.own_terms)   # df ignores document-title-only occurrences
        # BM25 idf with the usual +1 smoothing so rare terms weigh more.
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def _term_score(self, t: str, tf: dict, dl: int) -> float:
        if t not in tf or t not in self.idf:
            return 0.0
        f = tf[t]
        return self.idf[t] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))

    def score(self, query_groups: list[list[str]], i: int) -> float:
        """Standard BM25, except each query word contributes its best variant only (no double counting)."""
        tf, dl = self.tf[i], len(self.passages[i].tokens)
        seen: set[tuple[str, ...]] = set()
        s = 0.0
        for g in query_groups:
            key = tuple(g)
            if key in seen:
                continue
            seen.add(key)
            s += max(self._term_score(t, tf, dl) for t in g)
        return s

    def retrieve(self, query: str, k: int = 4) -> list[tuple[Passage, float]]:
        q = tokenize_groups(query, expand_query=True)
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
