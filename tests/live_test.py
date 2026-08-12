"""Live test — run benchmarks against real LLM."""

import time
import os
import sys

sys.path.insert(0, ".")

# Load .env
with open(os.path.expanduser("~/.hermes/.env")) as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v

from cab.runners import LLMRunner
from cab.benchmarks.social import PrisonersDilemmaBenchmark
from cab.benchmarks.cognitive import TowerOfHanoiBenchmark
from cab.benchmarks.emergence import SimpleRulesComplexBehaviorBenchmark
from cab.core import AgentState
from cab.scorer import aggregate_results, save_results

key = os.environ.get("XIAOMI_API_KEY", "")
url = os.environ.get("XIAOMI_BASE_URL", "")

if not key:
    print("XIAOMI_API_KEY not set")
    sys.exit(1)

runner = LLMRunner(
    model="mimo-v2.5-pro",
    api_key=key,
    base_url=url,
    max_tokens=2000,
    temperature=0.7,
)

benchmarks_to_run = [
    PrisonersDilemmaBenchmark,
    SimpleRulesComplexBehaviorBenchmark,
]

results = []
for BenchCls in benchmarks_to_run:
    bench = BenchCls()
    print(f"\n{'='*60}")
    print(f"Running: {bench.name} ({bench.benchmark_id})")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        result = bench.run(runner, "mimo-v2.5-pro")
        elapsed = time.time() - t0
        results.append(result)
        print(f"  CAS: {result.cas:.3f}")
        print(f"  Fitness: {result.fitness:.3f}")
        print(f"  Efficiency: {result.efficiency:.3f}")
        print(f"  Emergence: {result.emergence:.3f}")
        print(f"  Robustness: {result.robustness:.3f}")
        print(f"  Fidelity: {result.fidelity:.3f}")
        print(f"  Turns: {len(result.turns)}")
        print(f"  Tokens: {result.total_tokens}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Completed: {result.completed}")
        if result.error:
            print(f"  Error: {result.error}")
    except Exception as e:
        print(f"  FAILED: {e}")
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")

if results:
    summary = aggregate_results(results)
    filepath = save_results(results, "mimo-v2.5-pro", "results")
    print(f"\n{'='*60}")
    print(f"Overall CAS: {summary['overall_cas']:.3f}")
    print(f"Completed: {summary['completed']}/{summary['count']}")
    print(f"Total Tokens: {summary['total_tokens']}")
    print(f"Results: {filepath}")
    print(f"{'='*60}")
