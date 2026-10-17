"""The RAG pipeline: ingest → embed → retrieve → generate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import ingestion
from .chunking import Chunk
from .config import settings
from .embeddings import Embedder, get_embedder
from .llm import LLMClient, get_llm
from .vector_store import NumpyVectorStore, SearchResult


@dataclass
class QueryResponse:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0


class RAGPipeline:
    """Coordinates chunking, embedding, retrieval, and generation.

    The TF-IDF embedder's vocabulary depends on the corpus, so whenever new
    documents arrive the pipeline re-fits the embedder and rebuilds the
    index from all known chunks. That is O(corpus) per ingest — fine for a
    reference implementation and for corpora of modest size; the embedding
    and store interfaces are the seams where a production system would swap
    in incremental indexing.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.embedder = embedder or get_embedder(settings.embedder)
        if settings.embedder == "sentence-transformers":
            self.embedder = get_embedder(
                "sentence-transformers", model_name=settings.sentence_transformer_model
            )
        self.llm = llm or (
            get_llm("openai", model=settings.openai_model)
            if settings.llm_backend == "openai"
            else get_llm("extractive")
        )
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )
        self.store = NumpyVectorStore()
        self._all_chunks: list[Chunk] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_texts(self, documents: list[tuple[str, str]]) -> int:
        """Ingest ``(source, text)`` pairs. Returns the number of new chunks."""
        new_chunks: list[Chunk] = []
        for source, text in documents:
            new_chunks.extend(
                ingestion.ingest_text(
                    text,
                    source=source,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
            )
        if not new_chunks:
            return 0
        self._all_chunks.extend(new_chunks)
        self._rebuild_index()
        return len(new_chunks)

    def ingest_directory(self, directory: str | Path) -> int:
        """Ingest every .txt/.md file in a directory. Returns chunk count."""
        new_chunks = ingestion.ingest_directory(
            directory,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not new_chunks:
            return 0
        self._all_chunks.extend(new_chunks)
        self._rebuild_index()
        return len(new_chunks)

    def _rebuild_index(self) -> None:
        texts = [c.text for c in self._all_chunks]
        self.embedder.fit(texts)
        embeddings = self.embedder.embed(texts)
        self.store.clear()
        self.store.add(self._all_chunks, embeddings)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        if len(self.store) == 0:
            return []
        k = top_k if top_k is not None else settings.default_top_k
        query_vector = self.embedder.embed([question])[0]
        return self.store.search(query_vector, top_k=k)

    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        start = time.perf_counter()
        results = self.retrieve(question, top_k=top_k)
        answer = self.llm.generate(question, results)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return QueryResponse(
            question=question,
            answer=answer,
            sources=[
                {
                    "source": r.chunk.source,
                    "chunk_index": r.chunk.chunk_index,
                    "score": round(r.score, 4),
                    "text": r.chunk.text,
                }
                for r in results
            ],
            latency_ms=round(latency_ms, 2),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_index(self, directory: str | Path) -> None:
        self.store.save(directory)

    @property
    def document_count(self) -> int:
        return len({c.source for c in self._all_chunks})

    @property
    def chunk_count(self) -> int:
        return len(self._all_chunks)
