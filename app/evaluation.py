"""Evaluation harness for the RAG pipeline.

Each evaluation case provides a question, a reference answer, the source
document(s) a correct retrieval should surface, and optional keywords. For
every case the harness runs the live pipeline and computes:

* **context_recall** — fraction of expected sources present in the
  retrieved chunks.
* **context_precision** — fraction of retrieved chunks that come from an
  expected source (a chunk-level relevance proxy).
* **answer_relevance** — token-level F1 between the generated answer and
  the reference answer.
* **groundedness** — fraction of the answer's content tokens that also
  appear in the retrieved context (a simple faithfulness proxy).
* **latency** — end-to-end milliseconds per query (mean and p95).

These are lightweight, deterministic checks — useful as regression signals
in CI and as a baseline before adopting heavier model-graded evaluation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .rag import RAGPipeline

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on",
    "for", "and", "or", "with", "by", "at", "from", "that", "this", "it",
    "as", "be", "can", "do", "does", "how", "what", "when", "who",
}


def _content_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _token_f1(predicted: str, reference: str) -> float:
    pred_tokens = _content_tokens(predicted)
    ref_tokens = _content_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_set, ref_set = set(pred_tokens), set(ref_tokens)
    overlap = len(pred_set & ref_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_set)
    recall = overlap / len(ref_set)
    return 2 * precision * recall / (precision + recall)


@dataclass
class EvalCase:
    question: str
    reference_answer: str
    expected_sources: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict) -> "EvalCase":
        return cls(
            question=raw["question"],
            reference_answer=raw.get("reference_answer", ""),
            expected_sources=raw.get("expected_sources", []),
            keywords=raw.get("keywords", []),
        )


@dataclass
class CaseResult:
    question: str
    context_precision: float
    context_recall: float
    answer_relevance: float
    groundedness: float
    latency_ms: float
    retrieved_sources: list[str]


def load_dataset(path: str | Path) -> list[EvalCase]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [EvalCase.from_dict(item) for item in raw]


def evaluate_case(pipeline: RAGPipeline, case: EvalCase, top_k: int = 4) -> CaseResult:
    response = pipeline.query(case.question, top_k=top_k)
    retrieved_sources = [s["source"] for s in response.sources]
    unique_retrieved = list(dict.fromkeys(retrieved_sources))

    expected = set(case.expected_sources)
    if expected:
        hits = expected & set(unique_retrieved)
        recall = len(hits) / len(expected)
        relevant_chunks = sum(1 for s in retrieved_sources if s in expected)
        precision = relevant_chunks / len(retrieved_sources) if retrieved_sources else 0.0
    else:
        recall, precision = 0.0, 0.0

    context_text = " ".join(s["text"] for s in response.sources)
    context_tokens = set(_content_tokens(context_text))
    answer_tokens = _content_tokens(response.answer)
    grounded = (
        sum(1 for t in answer_tokens if t in context_tokens) / len(answer_tokens)
        if answer_tokens
        else 0.0
    )

    return CaseResult(
        question=case.question,
        context_precision=round(precision, 4),
        context_recall=round(recall, 4),
        answer_relevance=round(_token_f1(response.answer, case.reference_answer), 4),
        groundedness=round(grounded, 4),
        latency_ms=response.latency_ms,
        retrieved_sources=unique_retrieved,
    )


def run_evaluation(
    pipeline: RAGPipeline,
    dataset: list[EvalCase],
    top_k: int = 4,
) -> dict:
    """Run all cases and return an aggregate report dictionary."""
    results = [evaluate_case(pipeline, case, top_k=top_k) for case in dataset]
    latencies = np.array([r.latency_ms for r in results], dtype=np.float64)

    def _mean(attr: str) -> float:
        return round(float(np.mean([getattr(r, attr) for r in results])), 4) if results else 0.0

    return {
        "num_cases": len(results),
        "top_k": top_k,
        "metrics": {
            "context_precision": _mean("context_precision"),
            "context_recall": _mean("context_recall"),
            "answer_relevance": _mean("answer_relevance"),
            "groundedness": _mean("groundedness"),
        },
        "latency_ms": {
            "mean": round(float(latencies.mean()), 2) if results else 0.0,
            "p95": round(float(np.percentile(latencies, 95)), 2) if results else 0.0,
        },
        "cases": [
            {
                "question": r.question,
                "context_precision": r.context_precision,
                "context_recall": r.context_recall,
                "answer_relevance": r.answer_relevance,
                "groundedness": r.groundedness,
                "latency_ms": r.latency_ms,
                "retrieved_sources": r.retrieved_sources,
            }
            for r in results
        ],
    }
