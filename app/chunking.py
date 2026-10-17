"""Text chunking for document ingestion.

The splitter works in two stages:

1. Split the raw text into paragraphs (blank-line separated).
2. Greedily pack paragraphs into chunks of at most ``chunk_size`` characters,
   hard-splitting any single paragraph that is longer than the limit, and
   carrying a trailing overlap into the next chunk so context is not lost at
   chunk boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single retrievable unit of text with provenance metadata."""

    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    return [p.strip() for p in parts if p.strip()]


def _hard_split(paragraph: str, chunk_size: int) -> list[str]:
    """Split an over-long paragraph on sentence boundaries where possible."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                pieces.append(current)
                current = ""
            for i in range(0, len(sentence), chunk_size):
                pieces.append(sentence[i : i + chunk_size])
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks tagged with ``source``."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    units: list[str] = []
    for paragraph in _split_paragraphs(text):
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
        else:
            units.extend(_hard_split(paragraph, chunk_size))

    raw_chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            raw_chunks.append(current)
            # Carry an overlap tail into the next chunk, trimmed so the new
            # chunk still respects the size limit.
            overlap_text = ""
            if chunk_overlap:
                room = chunk_size - len(unit) - 2  # 2 for the "\n\n" joiner
                if room > 0:
                    overlap_text = current[-min(chunk_overlap, room) :]
            current = f"{overlap_text}\n\n{unit}".strip() if overlap_text else unit
        else:
            current = unit
    if current:
        raw_chunks.append(current)

    return [
        Chunk(
            text=chunk,
            source=source,
            chunk_index=i,
            metadata=dict(metadata or {}),
        )
        for i, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]
