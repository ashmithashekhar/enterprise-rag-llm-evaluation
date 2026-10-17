#!/usr/bin/env python3
"""Run the evaluation harness against the bundled synthetic sample data.

Usage:
    python scripts/run_evaluation.py [--top-k 4] [--out report.json]

Everything runs locally with the default TF-IDF embedder and the extractive
answer backend — no API key required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation import load_dataset, run_evaluation  # noqa: E402
from app.rag import RAGPipeline  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--out", type=str, default="", help="Optional path to save the JSON report")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    chunks = pipeline.ingest_directory(BASE_DIR / "data" / "sample_documents")
    print(f"Ingested {chunks} chunks from {pipeline.document_count} sample documents.")

    dataset = load_dataset(BASE_DIR / "data" / "eval_dataset.json")
    report = run_evaluation(pipeline, dataset, top_k=args.top_k)

    print(f"\nEvaluation over {report['num_cases']} cases (top_k={report['top_k']}):")
    for name, value in report["metrics"].items():
        print(f"  {name:<20} {value:.4f}")
    print(f"  latency mean (ms)    {report['latency_ms']['mean']:.2f}")
    print(f"  latency p95  (ms)    {report['latency_ms']['p95']:.2f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nFull report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
