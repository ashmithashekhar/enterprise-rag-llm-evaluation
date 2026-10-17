"""FastAPI service exposing the RAG pipeline and evaluation harness.

Run locally with:

    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive API docs.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .evaluation import EvalCase, load_dataset, run_evaluation
from .rag import RAGPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = BASE_DIR / "data" / "sample_documents"
SAMPLE_EVAL_PATH = BASE_DIR / "data" / "eval_dataset.json"

app = FastAPI(
    title="Enterprise RAG & LLM Evaluation Platform",
    description=(
        "Retrieval-augmented generation with a built-in evaluation harness. "
        "Runs fully offline with TF-IDF embeddings and an extractive answer "
        "backend — no API key required."
    ),
    version="0.1.0",
)

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------
class DocumentIn(BaseModel):
    source: str = Field(..., description="Document name / identifier")
    text: str = Field(..., description="Full document text")


class IngestRequest(BaseModel):
    documents: list[DocumentIn]
    load_sample_data: bool = Field(
        False, description="Also ingest the bundled synthetic sample documents"
    )


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int
    total_documents: int


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceOut(BaseModel):
    source: str
    chunk_index: int
    score: float
    text: str


class QueryResponseOut(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]
    latency_ms: float


class EvalCaseIn(BaseModel):
    question: str
    reference_answer: str = ""
    expected_sources: list[str] = []
    keywords: list[str] = []


class EvaluateRequest(BaseModel):
    dataset: list[EvalCaseIn] | None = Field(
        None,
        description="Inline eval cases. If omitted, the bundled synthetic dataset is used.",
    )
    top_k: int = 4


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "documents": pipeline.document_count,
        "chunks": pipeline.chunk_count,
        "embedder": settings.embedder,
        "llm_backend": settings.llm_backend,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    pipeline = get_pipeline()
    added = 0
    if request.documents:
        added += pipeline.ingest_texts(
            [(doc.source, doc.text) for doc in request.documents]
        )
    if request.load_sample_data:
        added += pipeline.ingest_directory(SAMPLE_DOCS_DIR)
    if added == 0:
        raise HTTPException(
            status_code=400,
            detail="No ingestible content provided (empty documents and no sample data requested).",
        )
    return IngestResponse(
        chunks_added=added,
        total_chunks=pipeline.chunk_count,
        total_documents=pipeline.document_count,
    )


@app.post("/query", response_model=QueryResponseOut)
def query(request: QueryRequest) -> QueryResponseOut:
    pipeline = get_pipeline()
    if pipeline.chunk_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Call /ingest first.",
        )
    response = pipeline.query(request.question, top_k=request.top_k)
    return QueryResponseOut(
        question=response.question,
        answer=response.answer,
        sources=[SourceOut(**s) for s in response.sources],
        latency_ms=response.latency_ms,
    )


@app.post("/evaluate")
def evaluate(request: EvaluateRequest) -> dict:
    pipeline = get_pipeline()
    if pipeline.chunk_count == 0:
        # Convenience for demos/tests: fall back to the bundled samples.
        pipeline.ingest_directory(SAMPLE_DOCS_DIR)
    if request.dataset is not None:
        cases = [
            EvalCase(
                question=c.question,
                reference_answer=c.reference_answer,
                expected_sources=c.expected_sources,
                keywords=c.keywords,
            )
            for c in request.dataset
        ]
    else:
        cases = load_dataset(SAMPLE_EVAL_PATH)
    return run_evaluation(pipeline, cases, top_k=request.top_k)
