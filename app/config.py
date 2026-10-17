"""Runtime configuration, overridable via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Retrieval
    default_top_k: int = int(os.getenv("TOP_K", "4"))

    # Embeddings backend: "tfidf" (default, no downloads) or "sentence-transformers"
    embedder: str = os.getenv("EMBEDDER", "tfidf")
    sentence_transformer_model: str = os.getenv(
        "ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # LLM backend: "extractive" (default, no API key needed) or "openai"
    llm_backend: str = os.getenv("LLM_BACKEND", "extractive")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Persistence directory for the vector store (empty string = in-memory only)
    index_dir: str = os.getenv("INDEX_DIR", "")


settings = Settings()
