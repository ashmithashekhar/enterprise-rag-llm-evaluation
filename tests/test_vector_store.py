import numpy as np

from app.chunking import Chunk
from app.embeddings import TfidfEmbedder
from app.vector_store import NumpyVectorStore


def _build_store():
    chunks = [
        Chunk(text="Cats are small domesticated mammals kept as pets.", source="cats.txt", chunk_index=0),
        Chunk(text="Stock markets rallied after the quarterly earnings report.", source="stocks.txt", chunk_index=0),
        Chunk(text="The recipe calls for flour, eggs, sugar and butter.", source="recipe.txt", chunk_index=0),
    ]
    embedder = TfidfEmbedder()
    embedder.fit([c.text for c in chunks])
    store = NumpyVectorStore()
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    return store, embedder


def test_search_returns_most_relevant_chunk_first():
    store, embedder = _build_store()
    results = store.search(embedder.embed(["What animals are kept as pets?"])[0], top_k=2)
    assert results
    assert results[0].chunk.source == "cats.txt"
    assert results[0].score >= results[1].score


def test_search_empty_store():
    store = NumpyVectorStore()
    assert store.search(np.zeros(3, dtype=np.float32)) == []


def test_persistence_round_trip(tmp_path):
    store, embedder = _build_store()
    store.save(tmp_path)
    loaded = NumpyVectorStore.load(tmp_path)
    assert len(loaded) == len(store)
    assert [c.text for c in loaded.chunks] == [c.text for c in store.chunks]
    results = loaded.search(embedder.embed(["flour eggs sugar butter recipe"])[0], top_k=1)
    assert results[0].chunk.source == "recipe.txt"
