# Flight Safety Intelligence Copilot

A retrieval-augmented question-answering system over NASA ASRS (Aviation Safety Reporting System) incident narratives, built on AWS Bedrock. Ask a natural-language question about air carrier safety events and get a grounded answer with the source reports that support it. Delay and on-time performance questions are routed to a separate BTS statistics path instead of the RAG pipeline. An LLM-as-judge, calibrated against hand-graded examples, scores answers for grounding without requiring a human to re-check every run.

The emphasis is on production concerns rather than a demo: a repeatable eval harness, hand-graded grounding results, explicit hallucination tests, and several documented gaps found via evals and fixed in place.

## Architecture

```
ASRS CSV export (4,584 air carrier narratives, Jan 2025 – Aug 2026)
        │
        ▼
  src/ingest.py ── Bedrock Titan Text Embeddings v2 ──▶ ChromaDB (local, persistent)
                                                              │
   question ──▶ is it a delay/on-time question? ──yes──▶ src/bts_stats.py ──▶ answer (no LLM call)
                                │no
                                ▼
                          embed ──▶ top-k similarity search ◀── ChromaDB
                                │
                                ▼
                   src/query.py ── Claude Sonnet 4.5 on Bedrock ──▶ answer + retrieved chunks
                                │
                                ▼
                   src/judge.py ── 3-vote LLM judge ──▶ grounding grade (y / partial / n)
```

- **Corpus:** NASA ASRS database export, filtered to *Reporter Organization = Air Carrier*, incident dates January 2025 through August 2026. Narratives are embedded as retrieval units.
- **Embeddings:** `amazon.titan-embed-text-v2:0`
- **Vector store:** ChromaDB, persisted to `data/chroma_db/`
- **Generation:** `us.anthropic.claude-sonnet-4-5-20250929-v1:0` via Bedrock inference profile, with exponential-backoff retry
- **Grounding contract:** the model is instructed to answer only from retrieved context, to say so when the context doesn't cover the question, and not to classify events against formal terms (e.g. "incursion") unless the source text itself uses that term
- **Airport metadata filtering:** the ASRS locale field is parsed into an `airport` tag at ingest; `query.py` detects a known airport code in the question and applies a ChromaDB `where` filter, widening `k` to cover the full airport-specific candidate set
- **Query routing:** delay/on-time/cancellation questions are detected by a topic-word + metric-word rule and answered directly from BTS on-time performance data (`src/bts_stats.py`), bypassing the RAG path entirely — no LLM call, no hallucination risk for these
- **LLM-as-judge:** `src/judge.py` grades each answer against its source chunks (or, for `bts_stats` rows, against the computed figures) using a rubric with few-shot examples drawn from this project's own hand-graded cases, and takes a majority vote across 3 independent calls per question

## Eval results

20 questions in `evals/questions.json`: 17 answerable (16 from ASRS, 1 from BTS stats), 3 deliberately out of scope (`should_answer: false`) to test refusal behavior. Two questions (`q011`, `q017`) name ATL; `q011` triggers the ASRS airport metadata filter, `q017` routes to BTS stats. Every answer was hand-graded against its source, and the LLM judge's grades were checked against the hand grades to calibrate it.

| Metric | Result |
|---|---|
| Fully grounded (`y`) | 17 / 20 |
| Partially grounded (`partial`) | 3 / 20 |
| Hallucinated claims (`n`) | 0 / 20 |
| Out-of-scope questions correctly refused | 3 / 3 |

The three `partial` grades, all found through iteration on this eval set rather than assumed at the outset:
- **q004** — a lightning-strike answer attributes weather-radar loss to the wrong incident; the detail belongs to a separate chunk describing a different flight.
- **q005** — an answer reverses causality, describing an ATC descent instruction as "complicating" a traffic conflict, when the source shows the crew was already descending on a TCAS Resolution Advisory before ATC's instruction.
- **q013** — an answer states a ground-crew procedural fault as settled fact, when the source only shows Ramp Control *reporting* that claim mid-call, during an already-degraded communication exchange with that same crew.

`q017` ("Delta's on-time rate at ATL") was a correct refusal before BTS ingestion and is now a grounded, non-LLM answer.

### Retrieval gap found and fixed via evals

`q011` asks about runway incursions *at ATL*. On the first run the retriever returned five ORD reports and the model correctly refused rather than fabricating — but two matching ATL reports existed in the corpus. Diagnosis: ATL has 10 tagged reports vs. ORD's 98, and the runway-incursion semantics outweighed the airport token in the embedding.

Fix: the ASRS `Locale Reference` field is now parsed into an `airport` metadata tag at ingest (974 of 4,584 reports carry a real code; the rest are de-identified as `ZZZ`), and `query.py` applies a ChromaDB `where` filter when a question names a known airport code. With the filter on, both ATL reports surface. Scoring the 10 ATL documents also showed cosine distances packed between 0.997 and 1.328 — evidence that pure vector similarity discriminates poorly among same-airport narratives, which motivates a reranker as a later step.

A second, subtler issue surfaced afterward: once retrieval was fixed, the model's own phrasing started denying that the (correctly retrieved) incursion evidence counted as an incursion, then, after a prompt fix, swung to under-stating the same evidence out of excess caution. Both are documented in `evals/eval_grades.csv`'s q011 note as a known limitation — retrieval found the right evidence in both cases, but the model's characterization of it wasn't yet reliable either way.

### Building and calibrating the LLM judge

A single LLM call per question, asked to grade its own kind of output, is not reliable on its own. A first version, evaluated in a single pass, agreed with the hand grades on only 14/20 (70%) questions — but on manual review, 3 of those 6 disagreements were judge hallucinations: it invented problems (a conflated chunk, a mischaracterized caveat, a misattributed procedural fault) that weren't actually in the text. Only 2 of the 6 disagreements were real bugs the judge had genuinely caught that the original hand grade had missed (q005, and eventually q013 on a later pass).

The fix was self-consistency: running the judge 3 times per question and taking a majority vote, rather than trusting any single call. Under voting, the noisy single-shot false positives (on q003, q011, q012) disappeared, while the two genuine catches (q005, q013) held up unanimously across all 3 votes on repeated runs. Every judge disagreement with a hand grade was independently re-verified against the source chunks before being accepted — the judge is a tool for flagging candidates for review, not a final authority run unsupervised.

## Repository layout

```
flight-copilot/
├── data/                  gitignored
│   ├── raw/               ASRS and BTS CSV exports
│   └── chroma_db/         persisted vector store
├── evals/
│   ├── questions.json     eval set with ids and should_answer flags
│   ├── eval_log.csv       answers + retrieved chunks from the last run
│   ├── eval_grades.csv    hand-graded grounding results with notes
│   └── judge_grades.csv   LLM judge's 3-vote grades and reasons
└── src/
    ├── paths.py           repo-relative path constants
    ├── ingest.py          CSV → embeddings → ChromaDB
    ├── query.py           retrieval + generation + BTS routing
    ├── bts_stats.py       BTS on-time performance lookups
    ├── judge.py           3-vote LLM judge, calibrated against hand grades
    ├── test_eval.py       runs every question, writes eval_log.csv
    ├── spot_check.py      interactive grader, writes eval_grades.csv
    └── inspect_data.py    ASRS export inspection helper
```

## Setup

Prerequisites: Python 3.11+, an AWS account with Bedrock model access enabled for Titan Embeddings v2 and Claude Sonnet 4.5 in `us-east-1`, and AWS credentials configured locally (`aws configure`).

```bash
git clone https://github.com/<your-username>/flight-copilot.git
cd flight-copilot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Get the ASRS data

1. Go to https://asrs.arc.nasa.gov/search/database.html
2. Add search items: *Reporter Organization was Air Carrier*, *Date of Incident between January-2025 and August-2026*
3. Run Search, then export as **Comma Separated File (CSV)**
4. Save as `data/raw/asrs_air_carrier_2025_2026.csv`

ASRS exports have a two-row header (category, field name). `ingest.py` reads it with `pd.read_csv(..., header=[0, 1])` and takes narrative text from `("Report 1", "Narrative")`, falling back to `("Report 1", "Synopsis")`.

### Get the BTS data

1. Go to https://www.transtats.bts.gov and navigate to **Airline Information for Download → Airline Service Quality Performance 234 (On-Time performance data)**
2. Filter Year: **2025**, Filter Period: **June**, Filter Geography: **All**
3. Check fields: `FlightDate`, `Reporting_Airline`, `Origin`, `Dest`, `DepDelayMinutes`, `ArrDelayMinutes`, `Cancelled`, `CancellationCode`
4. Download and save as `data/raw/bts_june_2025.csv`

Currently covers June 2025 only; `src/bts_stats.py` reads this single file. Adding more months means downloading additional CSVs and concatenating them before the stats functions will reflect a wider period.

### Build and run

```bash
python -m src.ingest              # embeds narratives, ~10 min for 4.5k rows
python -m src.test_eval           # runs all 20 questions → evals/eval_log.csv
python -m src.spot_check --n 20   # interactive hand-grading → evals/eval_grades.csv
python -m src.judge               # 3-vote LLM grading → evals/judge_grades.csv, plus agreement report
```

All scripts run as modules from the repo root and resolve paths via `src/paths.py`, so they work regardless of the current working directory.

## Gotchas encountered

- Newer Claude models on Bedrock require the inference-profile ID (`us.` prefix), not the bare model ID, for on-demand invocation.
- First-time Anthropic model use on Bedrock requires a use-case form; propagation across regions took 15–60 minutes and caused intermittent failures during that window, which is why `ask_claude()` retries with backoff.
- ASRS *Location* is only populated on a subset of reports; filtering on it in the ASRS query tool dropped the corpus from 15,692 to 56. Filter on date and reporter organization instead, and handle location at query time.
- BTS's raw monthly `.asc` download (from the "Airline Service Quality Performance" file list) is pipe-delimited with no header row and 40+ positional fields — not worth reverse-engineering. The TranStats field-selection tool at the same site exports a labeled CSV with only the chosen columns instead.
- A stats-question router keyed on bare words like "delay" produces false positives: "ground delay" is a common ASRS operational phrase, unrelated to BTS on-time statistics. Requiring both a topic word (on-time/cancellation) and a metric word (rate/percentage/performance) fixed it.
- A single LLM-judge call is not a reliable grader on its own — it hallucinates critiques at a meaningful rate. Self-consistency (majority vote across repeated calls) filtered out one-off noise, but every disagreement with the hand grades was still manually re-verified before being trusted.

## Roadmap

1. Reranking to sharpen retrieval within a single airport, where cosine distance alone discriminates poorly (see the ATL fix above)
2. Retrieval quality iteration: chunk sizing, hybrid search — measured against the eval set, not vibes
3. Cost and latency instrumentation per query
4. Airport-name → code mapping (e.g. "Hartsfield-Jackson" → ATL) so the filter fires on names, not just codes
5. Expand BTS coverage beyond June 2025 (additional months) and beyond ATL-only routing (other hub airports)
6. Judge quoting: require the judge to cite the exact source text behind any `partial`/`n` verdict, so its critiques are checkable rather than asserted, reducing reliance on manual spot-checks

## Data sources

- NASA Aviation Safety Reporting System — https://asrs.arc.nasa.gov. ASRS reports are voluntary, de-identified, and not a statistical sample of all events; findings from this tool describe what was reported, not incident rates.
- U.S. DOT Bureau of Transportation Statistics, Reporting Carrier On-Time Performance — https://www.transtats.bts.gov
