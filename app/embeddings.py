"""Pluggable text embedding backends.

Two backends are provided:

* ``TfidfEmbedder`` — the default. Pure scikit-learn TF-IDF; needs no model
  downloads and no API key, which makes the whole platform runnable offline.
  Its vocabulary is fit on the ingested corpus, so ``fit`` must be called
  before ``embed``.
* ``SentenceTransformerEmbedder`` — optional neural embeddings. Only used
  when ``sentence-transformers`` is installed and explicitly selected.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Minimal interface every embedding backend must satisfy."""

    @property
    def dimension(self) -> int: ...

    def fit(self, texts: list[str]) -> None: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class TfidfEmbedder:
    """TF-IDF embeddings backed by scikit-learn."""

    def __init__(self, max_features: int | None = 4096) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
        )
        self._fitted = False

    @property
    def dimension(self) -> int:
        if not self._fitted:
            raise RuntimeError("Embedder is not fitted yet — call fit() first.")
        return len(self._vectorizer.vocabulary_)

    def fit(self, texts: list[str]) -> None:
        if not texts:
            raise ValueError("Cannot fit TF-IDF embedder on an empty corpus.")
        self._vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder is not fitted yet — call fit() first.")
        matrix = self._vectorizer.transform(texts).toarray().astype(np.float32)
        return _l2_normalize(matrix)


class SentenceTransformerEmbedder:
    """Neural embeddings via sentence-transformers (optional dependency)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "sentence-transformers is not installed. Install it with "
                "`pip install sentence-transformers` or use EMBEDDER=tfidf."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def fit(self, texts: list[str]) -> None:  # noqa: ARG002 - pretrained model
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(name: str, **kwargs) -> Embedder:
    """Factory for embedding backends."""
    normalized = name.strip().lower()
    if normalized == "tfidf":
        return TfidfEmbedder(**kwargs)
    if normalized in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"Unknown embedder backend: {name!r}")
