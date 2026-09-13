# Flight Safety Intelligence Copilot

A retrieval-augmented question-answering system over NASA ASRS (Aviation Safety Reporting System) incident narratives, built on AWS Bedrock. Ask a natural-language question about air carrier safety events and get a grounded answer with the source reports that support it.

The emphasis is on production concerns rather than a demo: a repeatable eval harness, hand-graded grounding results, explicit hallucination tests, and a documented retrieval gap with a planned fix.

## Architecture

```
ASRS CSV export (4,584 air carrier narratives, Jan 2025 – Aug 2026)
        │
        ▼
  src/ingest.py ── Bedrock Titan Text Embeddings v2 ──▶ ChromaDB (local, persistent)
                                                              │
   question ──▶ embed ──▶ top-k similarity search ◀───────────┘
                                │
                                ▼
                   src/query.py ── Claude Sonnet 4.5 on Bedrock ──▶ answer + retrieved chunks
```

- **Corpus:** NASA ASRS database export, filtered to *Reporter Organization = Air Carrier*, incident dates January 2025 through August 2026. Narratives are embedded as retrieval units.
- **Embeddings:** `amazon.titan-embed-text-v2:0`
- **Vector store:** ChromaDB, persisted to `data/chroma_db/`
- **Generation:** `us.anthropic.claude-sonnet-4-5-20250929-v1:0` via Bedrock inference profile, with exponential-backoff retry
- **Grounding contract:** the model is instructed to answer only from retrieved context and to say so when the context doesn't cover the question

## Eval results

20 questions in `evals/questions.json`: 16 answerable from the corpus, 4 deliberately out of scope (`should_answer: false`) to test refusal behavior. Each answer was hand-graded against the retrieved chunks; grades and notes are in `evals/eval_grades.csv`.

| Metric | Result |
|---|---|
| Answerable questions fully grounded | 15 / 16 |
| Partially grounded | 1 / 16 (one soft inference about icing conditions not explicitly stated in source) |
| Hallucinated claims | 0 |
| Out-of-scope questions correctly refused | 4 / 4 |

Corpus size mattered: on an earlier 116-narrative corpus (Georgia-only filter), the turbulence question returned a single incident. On the 4,584-narrative corpus the same question, prompt, and model returned eight distinct incident categories, all traceable to source.

### Known retrieval gap

`q011` asks about runway incursions *at ATL*. The retriever returned five ORD reports and the model correctly refused rather than fabricating an ATL answer — but ATL reports do exist in the corpus. The runway-incursion semantics outweighed the airport token in the embedding. A neighboring question (`q017`) that named ATL without a strong topical signal *did* retrieve ATL reports, confirming the airport is embedded but not dominant. Fix in progress: attach the ASRS locale/airport field as chunk metadata at ingest and apply a ChromaDB `where` filter when a question names an airport.

## Repository layout

```
flight-copilot/
├── data/                  gitignored
│   ├── raw/               ASRS CSV exports
│   └── chroma_db/         persisted vector store
├── evals/
│   ├── questions.json     eval set with ids and should_answer flags
│   ├── eval_log.csv       answers + retrieved chunks from the last run
│   └── eval_grades.csv    hand-graded grounding results with notes
└── src/
    ├── paths.py           repo-relative path constants
    ├── ingest.py          CSV → embeddings → ChromaDB
    ├── query.py           retrieval + generation
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

### Get the data

1. Go to https://asrs.arc.nasa.gov/search/database.html
2. Add search items: *Reporter Organization was Air Carrier*, *Date of Incident between January-2025 and August-2026*
3. Run Search, then export as **Comma Separated File (CSV)**
4. Save as `data/raw/asrs_air_carrier_2025_2026.csv`

ASRS exports have a two-row header (category, field name). `ingest.py` reads it with `pd.read_csv(..., header=[0, 1])` and takes narrative text from `("Report 1", "Narrative")`, falling back to `("Report 1", "Synopsis")`.

### Build and run

```bash
python -m src.ingest              # embeds narratives, ~10 min for 4.5k rows
python -m src.test_eval           # runs all 20 questions → evals/eval_log.csv
python -m src.spot_check --n 20   # interactive grading → evals/eval_grades.csv
```

All scripts run as modules from the repo root and resolve paths via `src/paths.py`, so they work regardless of the current working directory.

## Gotchas encountered

- Newer Claude models on Bedrock require the inference-profile ID (`us.` prefix), not the bare model ID, for on-demand invocation.
- First-time Anthropic model use on Bedrock requires a use-case form; propagation across regions took 15–60 minutes and caused intermittent failures during that window, which is why `ask_claude()` retries with backoff.
- ASRS *Location* is only populated on a subset of reports; filtering on it in the ASRS query tool dropped the corpus from 15,692 to 56. Filter on date and reporter organization instead, and handle location at query time.

## Roadmap

1. Airport metadata filtering to close the `q011` retrieval gap
2. LLM-as-judge scoring calibrated against the hand grades in `eval_grades.csv`, so evals can run unattended
3. Ingest DOT/BTS On-Time Performance data and add query routing so questions like "Delta's on-time rate at ATL" (currently a correct refusal) become answerable
4. Retrieval quality iteration: chunk sizing, hybrid search, reranking — measured against the eval set, not vibes
5. Cost and latency instrumentation per query

## Data source

NASA Aviation Safety Reporting System — https://asrs.arc.nasa.gov. ASRS reports are voluntary, de-identified, and not a statistical sample of all events; findings from this tool describe what was reported, not incident rates.
