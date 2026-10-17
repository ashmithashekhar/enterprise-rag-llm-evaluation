"""A small cosine-similarity vector store backed by NumPy.

The store keeps L2-normalized embeddings in a single matrix, so cosine
similarity is one matrix-vector product. For the corpus sizes this reference
implementation targets (thousands to tens of thousands of chunks) that is
plenty fast, it has zero extra dependencies, and it persists to disk with
``save`` / ``load``. For million-scale corpora, the same interface can be
backed by FAISS without changing callers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class NumpyVectorStore:
    """In-memory vector store with JSON + NPZ persistence."""

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks.")
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            if embeddings.shape[1] != self._embeddings.shape[1]:
                raise ValueError("Embedding dimension mismatch with existing index.")
            self._embeddings = np.vstack([self._embeddings, embeddings])
        self._chunks.extend(chunks)

    def clear(self) -> None:
        self._embeddings = None
        self._chunks = []

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        if self._embeddings is None or len(self._chunks) == 0:
            return []
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        scores = self._embeddings @ query
        k = min(top_k, len(self._chunks))
        # argpartition for the top-k, then sort just those by score descending
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [
            SearchResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_indices
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._embeddings is None:
            np.savez(directory / "embeddings.npz", embeddings=np.zeros((0, 0)))
        else:
            np.savez(directory / "embeddings.npz", embeddings=self._embeddings)
        with open(directory / "chunks.json", "w", encoding="utf-8") as fh:
            json.dump([c.to_dict() for c in self._chunks], fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "NumpyVectorStore":
        directory = Path(directory)
        store = cls()
        with np.load(directory / "embeddings.npz") as data:
            embeddings = data["embeddings"]
        with open(directory / "chunks.json", encoding="utf-8") as fh:
            raw_chunks = json.load(fh)
        chunks = [
            Chunk(
                text=c["text"],
                source=c["source"],
                chunk_index=c["chunk_index"],
                metadata=c.get("metadata", {}),
            )
            for c in raw_chunks
        ]
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Persisted index is corrupt: chunk/embedding count mismatch.")
        if chunks:
            store._embeddings = embeddings.astype(np.float32)
            store._chunks = chunks
        return store
