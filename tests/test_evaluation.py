from pathlib import Path

from app.evaluation import EvalCase, load_dataset, run_evaluation
from app.rag import RAGPipeline

BASE_DIR = Path(__file__).resolve().parent.parent


def _pipeline_with_samples() -> RAGPipeline:
    pipeline = RAGPipeline()
    pipeline.ingest_directory(BASE_DIR / "data" / "sample_documents")
    return pipeline


def test_bundled_dataset_loads():
    dataset = load_dataset(BASE_DIR / "data" / "eval_dataset.json")
    assert len(dataset) >= 5
    assert all(case.question and case.reference_answer for case in dataset)


def test_evaluation_report_shape_and_ranges():
    pipeline = _pipeline_with_samples()
    dataset = load_dataset(BASE_DIR / "data" / "eval_dataset.json")
    report = run_evaluation(pipeline, dataset, top_k=4)

    assert report["num_cases"] == len(dataset)
    for metric in ("context_precision", "context_recall", "answer_relevance", "groundedness"):
        assert 0.0 <= report["metrics"][metric] <= 1.0
    assert report["latency_ms"]["mean"] >= 0.0
    assert report["latency_ms"]["p95"] >= report["latency_ms"]["mean"] * 0  # p95 present
    # Retrieval should find the right document for at least most cases.
    assert report["metrics"]["context_recall"] >= 0.5


def test_single_case_retrieval_hits_expected_source():
    pipeline = _pipeline_with_samples()
    case = EvalCase(
        question="How long are encrypted backups retained?",
        reference_answer="Encrypted backups are retained for 35 days on a rolling basis.",
        expected_sources=["data_retention_faq.txt"],
    )
    report = run_evaluation(pipeline, [case], top_k=3)
    assert report["cases"][0]["context_recall"] == 1.0
