"""Document ingestion: load raw text / files and turn them into chunks."""

from __future__ import annotations

from pathlib import Path

from .chunking import Chunk, chunk_text

SUPPORTED_SUFFIXES = {".txt", ".md"}


def ingest_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Chunk a single in-memory document."""
    return chunk_text(
        text,
        source=source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata=metadata,
    )


def ingest_file(
    path: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Chunk a single .txt / .md file from disk."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type {path.suffix!r}; supported: {sorted(SUPPORTED_SUFFIXES)}"
        )
    text = path.read_text(encoding="utf-8")
    return chunk_text(
        text,
        source=path.name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata={"path": str(path)},
    )


def ingest_directory(
    directory: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Chunk every supported file in a directory (non-recursive)."""
    directory = Path(directory)
    chunks: list[Chunk] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            chunks.extend(
                ingest_file(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            )
    return chunks
