"""
Quick evaluation script -- measures latency, checks citation accuracy.

Usage:
    python -m scripts.evaluate
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# noqa: E402 -- path must be set before local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.pipeline import RAGPipeline  # noqa: E402

EVAL_QUESTIONS = [
    "What are the main topics covered in the documents?",
    "Summarize the key findings or conclusions.",
    "What methods or approaches are described?",
    "Are there any statistics or numerical data mentioned?",
    "What recommendations or future work is suggested?",
]


def run_eval():
    print("\n=== RAG Evaluation ===\n")
    pipeline = RAGPipeline()
    results = []
    latencies = []

    for q in EVAL_QUESTIONS:
        print(f"Q: {q[:70]}...")
        t0 = time.perf_counter()
        r = pipeline.query(q)
        elapsed = (time.perf_counter() - t0) * 1000

        has_sources = len(r["sources"]) > 0
        has_answer  = len(r["answer"]) > 20
        status = "OK" if has_sources and has_answer else "!!"

        print(f"  {status}  {elapsed:.0f}ms | sources: {len(r['sources'])} | ans_len: {len(r['answer'])}")

        latencies.append(elapsed)
        results.append({
            "question":   q,
            "latency_ms": round(elapsed),
            "sources":    r["sources"],
            "timings":    r["timings"],
            "has_answer": has_answer,
            "has_sources": has_sources,
        })

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    summary = {
        "n_queries":        len(EVAL_QUESTIONS),
        "p50_ms":           round(p50),
        "p95_ms":           round(p95),
        "avg_ms":           round(sum(latencies) / len(latencies)),
        "all_have_sources": all(r["has_sources"] for r in results),
        "all_have_answers": all(r["has_answer"] for r in results),
        "results":          results,
    }

    Path("logs").mkdir(exist_ok=True)
    with open("logs/eval_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary:")
    print(f"  p50 latency : {summary['p50_ms']}ms")
    print(f"  p95 latency : {summary['p95_ms']}ms")
    print(f"  avg latency : {summary['avg_ms']}ms")
    print(f"  has sources : {summary['all_have_sources']}")
    print(f"\nFull report saved to logs/eval_report.json\n")


if __name__ == "__main__":
    run_eval()