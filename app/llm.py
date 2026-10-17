"""Pluggable answer-generation backends.

The default backend is extractive: it composes an answer from the retrieved
context itself (the sentences most relevant to the question), which means the
platform runs end-to-end with no API key and answers are always grounded in
retrieved text. An OpenAI backend is provided behind the same interface for
deployments that have a key.
"""

from __future__ import annotations

import os
import re
from typing import Protocol, runtime_checkable

from .vector_store import SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface every generation backend must satisfy."""

    def generate(self, question: str, context: list[SearchResult]) -> str: ...


class ExtractiveLLM:
    """Answer by extracting the most question-relevant context sentences.

    Sentences from the retrieved chunks are scored by token overlap with the
    question and the best few are stitched together with source citations.
    This is deliberately simple and transparent — a dependable no-key
    fallback and a useful baseline for evaluation.
    """

    def __init__(self, max_sentences: int = 4) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, context: list[SearchResult]) -> str:
        if not context:
            return "I don't have enough information in the provided documents to answer that."

        question_tokens = set(_tokens(question))
        scored: list[tuple[float, int, str, str]] = []
        seen: set[str] = set()
        for rank, result in enumerate(context):
            sentences = re.split(r"(?<=[.!?])\s+", result.chunk.text.replace("\n", " "))
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 25 or sentence.lower() in seen:
                    continue
                seen.add(sentence.lower())
                overlap = len(question_tokens & set(_tokens(sentence)))
                # rank bonus keeps higher-retrieved chunks slightly ahead on ties
                score = overlap + (len(context) - rank) * 0.01
                scored.append((score, rank, sentence, result.chunk.source))

        if not scored:
            best = context[0]
            return f"According to {best.chunk.source}: {best.chunk.text.strip()}"

        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = sorted(scored[: self.max_sentences], key=lambda item: item[1])
        parts = [f"{sentence} [{source}]" for _, _, sentence, source in chosen]
        return " ".join(parts)


class OpenAIClient:
    """Generation via the OpenAI API (optional dependency + API key)."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "openai is not installed. Install it with `pip install openai` "
                "or use LLM_BACKEND=extractive."
            ) from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI()
        self._model = model

    def generate(self, question: str, context: list[SearchResult]) -> str:
        context_text = "\n\n".join(
            f"[{r.chunk.source}] {r.chunk.text}" for r in context
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the provided context. "
                        "Cite sources inline like [source]. If the context does "
                        "not contain the answer, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {question}",
                },
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


def get_llm(backend: str, **kwargs) -> LLMClient:
    """Factory for generation backends."""
    normalized = backend.strip().lower()
    if normalized == "extractive":
        return ExtractiveLLM(**kwargs)
    if normalized == "openai":
        return OpenAIClient(**kwargs)
    raise ValueError(f"Unknown LLM backend: {backend!r}")
