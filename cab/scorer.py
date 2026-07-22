"""Scoring and metrics aggregation."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from cab.core import BenchmarkResult


# CAS weights
DEFAULT_WEIGHTS = {
    "fitness": 0.30,
    "efficiency": 0.20,
    "emergence": 0.25,
    "robustness": 0.15,
    "fidelity": 0.10,
}


def compute_cas(result: BenchmarkResult, weights: Optional[dict] = None) -> float:
    """Compute Cognitive Automaton Score for a single benchmark."""
    w = weights or DEFAULT_WEIGHTS
    return result.compute_cas(w)


def aggregate_results(results: list[BenchmarkResult], weights: Optional[dict] = None) -> dict:
    """Aggregate multiple benchmark results into a summary."""
    if not results:
        return {"cas": 0.0, "count": 0}

    w = weights or DEFAULT_WEIGHTS

    scores = {
        "fitness": [],
        "efficiency": [],
        "emergence": [],
        "robustness": [],
        "fidelity": [],
        "cas": [],
    }

    by_category = {}
    by_difficulty = {}

    for r in results:
        r.compute_cas(w)
        scores["fitness"].append(r.fitness)
        scores["efficiency"].append(r.efficiency)
        scores["emergence"].append(r.emergence)
        scores["robustness"].append(r.robustness)
        scores["fidelity"].append(r.fidelity)
        scores["cas"].append(r.cas)

        cat = r.category.value
        by_category.setdefault(cat, []).append(r.cas)

        diff = r.difficulty.value
        by_difficulty.setdefault(diff, []).append(r.cas)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "count": len(results),
        "completed": sum(1 for r in results if r.completed),
        "overall_cas": round(avg(scores["cas"]), 4),
        "dimensions": {k: round(avg(v), 4) for k, v in scores.items()},
        "by_category": {k: round(avg(v), 4) for k, v in by_category.items()},
        "by_difficulty": {k: round(avg(v), 4) for k, v in by_difficulty.items()},
        "total_tokens": sum(r.total_tokens for r in results),
        "total_latency_ms": round(sum(r.total_latency_ms for r in results), 1),
    }


def save_results(
    results: list[BenchmarkResult],
    model: str,
    output_dir: str = "results",
) -> str:
    """Save results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model.replace('/', '_')}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    summary = aggregate_results(results)
    data = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "benchmarks": [r.to_dict() for r in results],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


def compare_models(result_sets: dict[str, list[BenchmarkResult]]) -> str:
    """Generate a comparison table across models."""
    rows = []
    for model, results in result_sets.items():
        summary = aggregate_results(results)
        rows.append({
            "Model": model,
            "CAS": summary["overall_cas"],
            "Fitness": summary["dimensions"]["fitness"],
            "Efficiency": summary["dimensions"]["efficiency"],
            "Emergence": summary["dimensions"]["emergence"],
            "Robustness": summary["dimensions"]["robustness"],
            "Fidelity": summary["dimensions"]["fidelity"],
            "Benchmarks": summary["count"],
            "Tokens": summary["total_tokens"],
        })

    # Sort by CAS descending
    rows.sort(key=lambda r: r["CAS"], reverse=True)

    from tabulate import tabulate
    return tabulate(rows, headers="keys", tablefmt="github", floatfmt=".4f")
