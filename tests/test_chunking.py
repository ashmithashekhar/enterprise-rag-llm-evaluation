import pytest

from app.chunking import chunk_text


def test_short_text_single_chunk():
    chunks = chunk_text("Hello world. This is a short document.", source="a.txt")
    assert len(chunks) == 1
    assert chunks[0].source == "a.txt"
    assert chunks[0].chunk_index == 0


def test_long_text_splits_within_size_limit():
    text = "\n\n".join(f"Paragraph {i}. " + "Some sample content here. " * 10 for i in range(12))
    chunks = chunk_text(text, source="b.txt", chunk_size=300, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_overlap_carries_context():
    text = "\n\n".join(f"Unique paragraph number {i} with enough words to fill space." for i in range(20))
    chunks = chunk_text(text, source="c.txt", chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    # The tail of one chunk should reappear at the start of the next.
    tail = chunks[0].text[-30:]
    assert tail[:20] in chunks[1].text


def test_empty_text_produces_no_chunks():
    assert chunk_text("   \n\n  ", source="empty.txt") == []


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text("some text", source="x.txt", chunk_size=100, chunk_overlap=100)
