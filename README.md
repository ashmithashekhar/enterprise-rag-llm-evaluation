# Enterprise RAG & LLM Evaluation Platform

A self-contained reference implementation of a **retrieval-augmented generation (RAG)** platform with a **built-in LLM evaluation harness**, exposed as a FastAPI service.

It is designed to run **fully offline with zero API keys**: document ingestion, chunking, TF-IDF embeddings, vector search, extractive answer generation, and evaluation all work out of the box. Optional backends (sentence-transformers embeddings, OpenAI generation) plug in behind the same interfaces when available.

> **Demo data note:** All bundled documents and evaluation cases are synthetic sample data about a fictional company ("Brightline Software"), written for this project. No real customer, employer, or proprietary data is included.

## Features

- **Document ingestion** — load `.txt` / `.md` files or raw text, split into overlapping chunks with source metadata preserved end-to-end.
- **Pluggable embeddings** — TF-IDF (default, scikit-learn, no downloads) or sentence-transformers for neural embeddings.
- **Vector store** — NumPy cosine-similarity index with save/load persistence; the interface is the seam where FAISS can be swapped in for very large corpora.
- **Pluggable generation** — extractive answer composer by default (answers are stitched from retrieved sentences with source citations), or OpenAI chat models when a key is configured.
- **Evaluation harness** — deterministic checks per case: context precision, context recall, answer relevance (token F1), groundedness, and latency (mean / p95), reported per case and in aggregate.
- **REST API** — FastAPI service with `/ingest`, `/query`, and `/evaluate` endpoints plus interactive OpenAPI docs.

## Architecture

```
┌────────────┐   ┌────────────┐   ┌──────────────┐   ┌─────────────┐
│  Documents │──▶│  Chunking  │──▶│  Embeddings  │──▶│ Vector Store│
│ .txt / .md │   │ (overlap)  │   │ TF-IDF / ST  │   │  (NumPy)    │
└────────────┘   └────────────┘   └──────────────┘   └──────┬──────┘
                                                            │ top-k
┌────────────┐   ┌────────────┐   ┌──────────────┐          │
│   Answer   │◀──│ Generation │◀──│  Retrieval   │◀─────────┘
│ + sources  │   │ extractive │   │ cosine sim.  │
└────────────┘   │ / OpenAI   │   └──────────────┘
                 └────────────┘
                        ▲
                 ┌──────────────┐
                 │  Evaluation  │  context precision/recall,
                 │   harness    │  answer relevance, groundedness,
                 └──────────────┘  latency — per case + aggregate
```

## Tech Stack

- **Python 3.11+**
- **FastAPI** + **Uvicorn** — API service
- **scikit-learn** — TF-IDF embeddings and text utilities
- **NumPy** — vector index and similarity search
- **Pydantic v2** — request/response schemas
- **pytest** — test suite
- Optional: **sentence-transformers**, **OpenAI API**, **FAISS**

## Setup

```bash
git clone <this-repo>
cd enterprise-rag-llm-evaluation

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### Run the API

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for interactive documentation.

```bash
# Ingest the bundled sample documents
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": [], "load_sample_data": true}'

# Ask a question
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How long are encrypted backups retained?"}'

# Run the bundled evaluation dataset
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Run evaluation from the command line

```bash
python scripts/run_evaluation.py --out eval_reports/report.json
```

This ingests the sample documents, runs the bundled synthetic eval dataset, prints aggregate metrics (context precision/recall, answer relevance, groundedness, latency), and optionally saves the full per-case report as JSON.

### Run the tests

```bash
pytest
```

## API Endpoints

| Method | Path        | Description                                                                 |
|--------|-------------|-----------------------------------------------------------------------------|
| GET    | `/health`   | Service status, index size, active backends                                 |
| POST   | `/ingest`   | Ingest documents (`{source, text}` list) and/or the bundled sample data     |
| POST   | `/query`    | Retrieve context and generate an answer with cited sources + latency        |
| POST   | `/evaluate` | Run eval cases (inline dataset, or the bundled one when omitted) and return the metrics report |

## Configuration

All settings are environment variables:

| Variable       | Default        | Purpose                                             |
|----------------|----------------|-----------------------------------------------------|
| `CHUNK_SIZE`   | `800`          | Max characters per chunk                            |
| `CHUNK_OVERLAP`| `120`          | Character overlap between consecutive chunks        |
| `TOP_K`        | `4`            | Default number of chunks retrieved per query        |
| `EMBEDDER`     | `tfidf`        | `tfidf` or `sentence-transformers`                  |
| `ST_MODEL`     | `all-MiniLM-L6-v2` | Sentence-transformers model name                |
| `LLM_BACKEND`  | `extractive`   | `extractive` (no key) or `openai`                   |
| `OPENAI_MODEL` | `gpt-4o-mini`  | Model used when `LLM_BACKEND=openai`                |
| `OPENAI_API_KEY` | —            | Required only for the OpenAI backend                |
| `INDEX_DIR`    | *(empty)*      | Reserved for persisted index location               |

Example with neural embeddings and OpenAI generation:

```bash
pip install sentence-transformers openai
EMBEDDER=sentence-transformers LLM_BACKEND=openai OPENAI_API_KEY=sk-... \
  uvicorn app.main:app
```

## Evaluation Metrics

| Metric              | Meaning                                                                 |
|---------------------|-------------------------------------------------------------------------|
| Context precision   | Fraction of retrieved chunks that come from an expected source document |
| Context recall      | Fraction of expected source documents represented in the retrieved set  |
| Answer relevance    | Token-level F1 between the generated answer and the reference answer    |
| Groundedness        | Fraction of answer content tokens present in the retrieved context      |
| Latency             | End-to-end query time, mean and p95 across cases (ms)                   |

These are lightweight deterministic checks — intended as fast regression signals and a baseline. A production deployment would typically add model-graded evaluation and human review on top.

## Project Structure

```
enterprise-rag-llm-evaluation/
├── app/
│   ├── main.py          # FastAPI application and endpoints
│   ├── rag.py           # RAGPipeline: ingest → retrieve → generate
│   ├── ingestion.py     # Document loading
│   ├── chunking.py      # Paragraph-aware overlapping chunk splitter
│   ├── embeddings.py    # TF-IDF and sentence-transformers backends
│   ├── vector_store.py  # NumPy cosine-similarity store with persistence
│   ├── llm.py           # Extractive and OpenAI generation backends
│   ├── evaluation.py    # Metrics and evaluation harness
│   └── config.py        # Environment-driven settings
├── data/
│   ├── sample_documents/  # Synthetic sample documents (fictional company)
│   └── eval_dataset.json  # Synthetic evaluation cases
├── scripts/
│   └── run_evaluation.py  # CLI evaluation runner
├── tests/                 # pytest suite (chunking, store, evaluation, API)
├── requirements.txt
└── README.md
```

## Limitations & Next Steps

- The TF-IDF index is rebuilt on each ingest (vocabulary depends on the corpus) — suitable for modest corpora; incremental indexing is the natural next step.
- The NumPy store is exact but linear-scan; FAISS or another ANN index would serve very large collections.
- The extractive generator is a transparent baseline, not a substitute for an LLM in production use.
- Possible extensions: hybrid (keyword + vector) retrieval, reranking, streaming responses, and model-graded evaluation.

## License

MIT — see repository for details. Sample data is synthetic and may be reused freely.
