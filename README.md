# AI Passenger Support Agent – Tel Aviv Red Line (demo)

A small, fully working portfolio project: a Hebrew text-and-voice support assistant for passengers of the
Tel Aviv Red Line light rail. It answers payment questions, explains what to do during a service
disruption, and helps at a station, **only from identified sources** (retrieved guidance passages and
simulated operational tools reached over MCP), and it recognises when a human representative is needed.

![Screenshot of the demo UI](docs/screenshot.png)

> **Everything operational in this project is demo data.** The guidance documents were written for this
> project and are not official operator documents. The service status and the support cases are simulated.
> The UI says so on every screen, and every reply carries a label saying how it was produced.

**Runs without an API key and without any language model.** The optional Claude integration is kept in the code
as a future capability and is off unless you add a key.

| Mode | Needs | What produces the reply |
|---|---|---|
| **Guided demo** (three cards) | Python only | A predefined text per scenario; retrieval and MCP calls are real |
| **Free questions, typed or spoken** (default) | Python only | A deterministic workflow: verbatim quotes of retrieved passages + real MCP results, assembled by code |
| Live mode (optional, costs API usage) | `ANTHROPIC_API_KEY` in `.env` | Claude decides which tool to call and writes the answer |

---

## 1. The passenger problem

A passenger standing at a gate with a card that will not validate, or on a platform wondering whether trains
are running, needs three different kinds of help at once:

| Need | Correct source | Wrong source |
|---|---|---|
| "What is the rule / what should I do?" | Written passenger guidance | General knowledge |
| "Is the line running *right now*?" | A live status system | Any document, however recent |
| "I want my money back / to appeal a fine" | A human representative | A chatbot promising an outcome |

A plain chatbot blurs these. This project keeps the three sources separate and shows its work on every reply.

## 2. Why RAG and MCP

- **RAG (retrieval-augmented generation)** is for *stable* knowledge: rules and procedures. Retrieving and citing
  passages makes a reply checkable and lets the system say "the documents do not cover this".
- **MCP (Model Context Protocol)** is for *actions and live data*: checking current service status and preparing
  a support case. These cannot live in documents and must be logged. Because they are MCP tools, the agent, the
  developer panel and the evaluation all see the same calls and results, and the demo server could be replaced by
  one that talks to a real operator system without touching the application.

The key rule: **documents may describe what to do during a disruption, but only the status tool may say
whether there is one.**

## 3. Architecture

```
Passenger (Hebrew; typed, or spoken via the browser's speech recognition, transcript editable before sending)
   │
   ├─ guided card ─► POST /api/offline/run      ├─ free question ─► POST /api/offline/ask     ├─ (live) POST /api/chat
   ▼                                            ▼                                             ▼
FastAPI app (app/server.py)
   │
   ├─ Retrieval ─── BM25 over docs/*.md split by "## " heading ──► top-4 passages with ids and scores
   │                (app/rag.py: Hebrew final-letter normalisation, prefix stripping, small query synonym list,
   │                 field weights: section title ×2, body ×1, document title ×0.25)
   │
   ├─ MCP client (app/mcp_client.py) ──stdio──► MCP server subprocess (mcp_server/server.py)
   │        get_service_status(station?)  ← reads demo_status.json (4 switchable scenarios, incl. a feed outage)
   │        prepare_support_case(...)     ← appends to demo_cases.json, returns DEMO-#### id
   │
   ├─ Reply assembly
   │     offline_demo.py      guided cards: scripted tool calls, predefined text filled from real tool results
   │     offline_workflow.py  free questions: regex rules decide live-status / handoff / clarify;
   │                          reply = quotes of passages scoring ≥ 6 + status sentence + case id; else "unsupported"
   │     agent.py (optional)  Claude tool loop; JSON {action, answer_he, sources}; citations validated in code
   │
   ▼
Answer card: label (predefined / templated / model), action badge, Hebrew reply, cited passages with excerpts,
             demo case id, "what actually happened" steps, ▶/■ text-to-speech controls
Developer panel: retrieved passages + scores, every MCP call with input/output, code flags, model output or
                 "no model call"
```

Actions: `answer` (grounded), `clarify` (one question back), `handoff` (a case was prepared through MCP),
`unsupported` (documents do not support an answer; nothing invented).

## 4. Layout

```
app/
  server.py          FastAPI app and JSON API (guided cards, free questions, optional chat, scenario switch)
  offline_demo.py    three guided scenarios (real retrieval + real MCP + predefined labelled replies)
  offline_workflow.py deterministic free-question workflow (rules + quotes + MCP; no model)
  agent.py           optional live mode (Claude tool loop + validation)
  prompts.py         system prompt for live mode
  rag.py             Markdown loader, heading chunker, Hebrew tokenizer, BM25
  mcp_client.py      spawns the MCP server, discovers tools, forwards calls, logs them
  config.py          paths, thresholds, model settings
  static/index.html  Hebrew RTL UI, speech recognition, text-to-speech, developer panel (vanilla JS)
mcp_server/          MCP server + demo_status.json (demo_cases.json is created at runtime, git-ignored)
docs/                7 demo guidance documents (Hebrew) + README + screenshot
eval/                cases.json (24 cases), run_eval.py, results.md (measured)
tests/               test_agent_offline.py (live agent loop with a stub model + real MCP server)
```

## 5. Run it locally (no API key)

Requirements: Python 3.10 or newer. Tested on Windows 11 with Python 3.10 and the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
```

```bash
python run.py
```

Open <http://127.0.0.1:8000>. The console shows:

```
[mcp-server] starting on stdio
[mcp-client] connected; tools: ['get_service_status', 'prepare_support_case']
[app] 31 passages indexed from ...\docs
[app] WARNING: no ANTHROPIC_API_KEY - chat endpoint will return an error
```

The header reads **"מצב: הדגמה ללא מודל"**. Every MCP call is printed as `[mcp-client] <tool>({...}) -> {...}`.

Without a browser:

```bash
python -m app.offline_demo
```

```bash
python -m eval.run_eval
```

```bash
python -m tests.test_agent_offline
```

Deep links for demos and screenshots: `/?autorun=payment_rag,status_mcp,handoff_mcp` runs the guided cards;
`/?ask=<question>` submits a free question.

## 6. Voice interface

- **Speech to text:** the 🎤 button uses the browser's Web Speech API (`SpeechRecognition`, language `he-IL`).
  The transcript is written into the text box while you speak; you can correct it and then press send. Where the
  API is missing, the button is disabled with an explanation and typing works as before.
- **Text to speech:** each reply has ▶ and ■ controls using `speechSynthesis` with a Hebrew (`he-*`) voice if one is
  installed. Without a Hebrew voice the play button is disabled and the reply says so; the text stays usable.
- A spoken question is treated exactly like a typed one: it goes through retrieval and the MCP workflow. It is not
  mapped to one of the guided cards.

Speech recognition in Chromium browsers is processed by the browser vendor's service, needs microphone permission,
and is not available in every browser. See section 8 for what was and was not verified.

## 7. Two-minute walkthrough for a recruiter

Start `python run.py`, open the page.

1. **Guided card 1 – payment (RAG).** Press "הרצת התרחיש" on *"אפשר לשלם במזומן ברכבת?"*. The developer panel shows
   four BM25 passages with scores; the cited one is green. No tool was called, and the panel says so.
2. **Guided card 2 – status (real MCP).** The scenario flips the simulated feed to "segment closed" and calls
   `get_service_status` for real; the panel shows the call and the JSON that came back, and the reply's first
   sentence is filled from it.
3. **Guided card 3 – handoff (real MCP).** `prepare_support_case` runs for real and returns an id such as
   `DEMO-0001`, shown in the reply and in the cases list.
4. **Ask your own question, typed or spoken.** For example *"המעלית באלנבי עובדת כרגע?"* after switching the
   feed to "elevator out". The workflow detects a live question and a station, calls the status tool, quotes the
   accessibility passage, and labels the reply as templated. Try *"לא עבד לי"* to see a clarifying question, and
   *"כמה זמן שומרים חפצים באבדות ומציאות?"* to see an honest "the document says this is unspecified".

What to say: the retrieval, the MCP tool discovery and calls, the citation validation and the logging are the
product; the reply text in offline mode is assembled by code and labelled as such.

## 8. What was measured

`eval/cases.json` holds 24 original cases (ordinary, live status incl. a feed outage, ambiguous incl. a two-turn
flow, missing information, conflicting documents, handoff, one safety case with a pasted card number) plus 10
holdout paraphrases marked `"holdout": true`. Checks are deterministic (actions, tools called or not, cited
documents, required or forbidden strings). No LLM judge.
Full output: [`eval/results.md`](eval/results.md).

| Part | What | Result (last run) |
|---|---|---|
| A | Retrieval: expected document in top 4 | 15/15 |
| B | Live agent (Claude) on 24 cases | **Not run** (no API key used in this project) |
| C | Guided cards: sources retrieved, MCP calls succeeded, case id returned | 3/3 |
| D | Deterministic free-question workflow on the original 24 cases (available while the rules were tuned) | 23/24 |
| E | Same workflow on 10 holdout paraphrases written afterwards, never used for tuning | 7/10 |

Part D failure, as measured: `amb_04` (two-turn follow-up "the credit card was not read at the gate"): the reply
quotes the "personal details" section of the escalation policy, which mentions credit cards, instead of the
"reader not responding" section, because the passenger's verb (נקלט) does not occur in that section.

Part E failures, as measured: `new_04` (a small dog on the train: the animals section scores below the quote
threshold, so the reply is an honest "unsupported"), `new_07` (a train that gets stuck: the reply quotes the
replacement-bus section rather than the "train stopped between stations" section, because נתקעת ≠ נעצרת), and
`new_08` ("I want my money back": no handoff rule covers this phrasing, so no case is prepared). Part E also
showed the live-status regex over-triggering on "בשעות הבוקר" in a bicycle question (`new_01` still passed).

Two earlier Part D failures were fixed with general rules rather than per-question patches: field-weighted
BM25 (section title ×2, body ×1, document title ×0.25, one score per question word) plus "quote only sections
whose own title contains a question word when such sections exist", and a price rule ("price question with no
passage stating a price → unsupported"). The eval assertions were not changed for these fixes.

Browser verification of the voice interface (Chromium 148 embedded browser, Windows 11, 2026-09-15):

- `SpeechRecognition` was present and the 🎤 button active. Pressing it requested the microphone, which that
  environment blocks; the page showed "שגיאת זיהוי דיבור: not-allowed – אפשר להקליד במקום" and typing kept working.
  **Real speech capture was therefore not exercised**; it needs a normal Chrome/Edge window with microphone permission.
- `speechSynthesis` was present with three English voices and no Hebrew voice, so ▶ was disabled with the note
  "אין קול עברי מותקן בדפדפן". Audio playback was not exercised.
- Three differently phrased questions were submitted through the same text box the transcript lands in and routed
  through the workflow: *"איך משלמים על הנסיעה, אפשר עם כרטיס אשראי?"* (answer, two payment passages quoted, no
  tool), *"יש עיכובים בקו כרגע? אני צריכה להגיע לאלנבי"* with the feed set to "segment closed" (real
  `get_service_status` call for אלנבי, closed segment and replacement bus reported), and *"חייבו אותי פעמיים על אותה
  נסיעה אתמול, מה עושים?"* (handoff, real `prepare_support_case` call, case `DEMO-0002`).

No business-impact figures are claimed.

## 9. Limitations

- **Offline replies are assembled by rules.** They quote documents verbatim and report tool results; they do not
  understand paraphrase. The threshold (`MIN_PASSAGE_SCORE = 6.0`) and the regex rules are visible in
  `app/offline_workflow.py` and were tuned on the original 24 cases, so Part D is not a blind test; Part E
  (holdout paraphrases) is the honest number, and its failures show the kinds of phrasing the rules miss.
- **Demo documents**, invented for the project; nothing in the code assumes their content except the eval cases,
  the guided scenarios and the conflict pair (`refunds_v2` vs `refunds_v1_old`).
- **Lexical retrieval** (BM25 with prefix stripping and a hand-made synonym list). Embeddings or hybrid retrieval
  would improve recall on free-form Hebrew.
- **Voice depends on the browser**: recognition needs a browser that implements the Web Speech API and microphone
  permission; Hebrew text-to-speech needs an installed Hebrew voice.
- **Single process, in-memory sessions**; simulated tools backed by JSON files; no streaming.
- **Live mode is unmeasured here.** Part B exists but was not run.

## 10. Decisions an interviewer may ask about

- **Why a deterministic offline path instead of a small local model?** The demo is about grounding: retrieval,
  MCP discovery and calls, validation, logging. Rules plus verbatim quotes keep every reply traceable and cost
  nothing; a weak local model would add unverifiable text.
- **Why a hand-written tool loop for live mode?** ~40 lines, fully visible; the validation after the loop is the point.
- **Why BM25?** 31 passages, Hebrew, explainability; every score is shown in the UI.
  Swapping in embeddings is a one-class change.
- **Why validate citations even for scripted replies?** Any reply can cite a passage retrieval never returned;
  dropping and flagging is cheaper than trust.
- **Why MCP for two small tools?** The app only knows tool names and schemas discovered at startup; replacing the
  demo server with one that calls a real status API changes no application code.

## 11. Optional live mode

Copy `.env.example` to `.env`, set a real `ANTHROPIC_API_KEY`, restart. The header switches to "מודל חי", the
text box routes to Claude (`claude-opus-5`, manual tool loop in `app/agent.py`), and `python -m eval.run_eval`
additionally runs Part B. This costs API usage; nothing else in the project does.
