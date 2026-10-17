from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingest_query_evaluate_flow():
    ingest = client.post(
        "/ingest",
        json={
            "documents": [
                {
                    "source": "garden.txt",
                    "text": (
                        "Tomatoes need full sun and regular watering. "
                        "Plant tomato seedlings after the last frost, spaced about 60 cm apart. "
                        "Mulching helps the soil retain moisture during dry weeks."
                    ),
                }
            ]
        },
    )
    assert ingest.status_code == 200
    assert ingest.json()["chunks_added"] >= 1

    query = client.post("/query", json={"question": "When should tomato seedlings be planted?"})
    assert query.status_code == 200
    body = query.json()
    assert body["answer"]
    assert body["sources"][0]["source"] == "garden.txt"
    assert body["latency_ms"] >= 0

    evaluation = client.post(
        "/evaluate",
        json={
            "dataset": [
                {
                    "question": "When should tomato seedlings be planted?",
                    "reference_answer": "Plant tomato seedlings after the last frost.",
                    "expected_sources": ["garden.txt"],
                }
            ],
            "top_k": 2,
        },
    )
    assert evaluation.status_code == 200
    report = evaluation.json()
    assert report["num_cases"] == 1
    assert report["cases"][0]["context_recall"] == 1.0


def test_query_before_ingest_is_rejected_on_fresh_pipeline(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "_pipeline", None)
    fresh_client = TestClient(main_module.app)
    response = fresh_client.post("/query", json={"question": "Anything ingested?"})
    assert response.status_code == 400
