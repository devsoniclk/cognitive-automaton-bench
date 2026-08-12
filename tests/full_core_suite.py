"""Full core suite run against mimo-v2.5-pro."""

import os
import sys
import time
import httpx
import json

sys.path.insert(0, ".")

# Load .env
with open(os.path.expanduser("~/.hermes/.env")) as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v

from cab.benchmarks import BENCHMARKS, SUITES
from cab.core import AgentState
from cab.scorer import aggregate_results, save_results

KEY = os.environ.get("XIAOMI_API_KEY", "")
URL = os.environ.get("XIAOMI_BASE_URL", "")
MODEL = "mimo-v2.5-pro"

if not KEY:
    print("XIAOMI_API_KEY not set"); sys.exit(1)


def call_mimo(system: str, user: str) -> str:
    resp = httpx.post(
        f"{URL}/chat/completions",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
        },
        timeout=60,
    )
    d = resp.json()
    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content:
        content = msg.get("reasoning_content") or ""
    return content


def run_benchmark(bench_id: str):
    """Run a single benchmark and return the result."""
    cls = BENCHMARKS[bench_id]
    bench = cls()
    env = bench.create_environment()
    const = bench.get_constitution()
    state = AgentState(resources=dict(env.resources))
    system = const.to_prompt() + "\n\n" + bench.build_system_prompt(env, state)

    from cab.core import BenchmarkResult, TurnResult, AgentAction

    result = BenchmarkResult(
        benchmark_id=bench_id,
        model=MODEL,
        category=bench.category,
        difficulty=bench.difficulty,
    )

    t_start = time.time()

    MAX_TURNS = 5  # Cap for live runs — reasoning model is slow

    for turn in range(min(bench.max_turns, MAX_TURNS)):
        if not state.alive:
            break
        env.tick = turn
        t0 = time.time()
        try:
            prompt = bench.build_turn_prompt(env, state, turn)
            resp = call_mimo(system, prompt)
            action, env = bench.process_response(resp, env, state, turn)
        except Exception as e:
            resp = ""
            action = AgentAction(action_type="error", reasoning=str(e))
            state.constitution_violations += 1
        elapsed = (time.time() - t0) * 1000
        state.actions_taken.append(action)
        print(f"    [{turn+1:2d}] {elapsed/1000:5.1f}s  {action.action_type}", flush=True)

        tr = TurnResult(
            turn=turn,
            agent_state=AgentState(
                agent_id=state.agent_id,
                alive=state.alive,
                resources=dict(state.resources),
                children=list(state.children),
                constitution_violations=state.constitution_violations,
            ),
            environment=env.snapshot(),
            action=action,
            response_raw=resp if "resp" in dir() else "",
            parsed_successfully=action.action_type not in ("error",),
            latency_ms=elapsed,
        )
        result.turns.append(tr)
        result.total_latency_ms += elapsed

        if action.action_type in ("terminate", "solved"):
            break

    # Score
    result.fitness = bench.score_fitness(state, env)
    result.efficiency = bench.score_efficiency(state, env)
    result.emergence = bench.score_emergence(state, env)
    result.robustness = bench.score_robustness(state, env)
    result.fidelity = bench.score_fidelity(state, env)
    result.compute_cas()
    result.completed = result.error is None
    result.total_latency_ms = (time.time() - t_start) * 1000

    return result


# Run all 8 core benchmarks
bench_ids = SUITES["core"]
print(f"{'='*70}")
print(f"  Cognitive Automaton Bench — Core Suite")
print(f"  Model: {MODEL}")
print(f"  Benchmarks: {len(bench_ids)}")
print(f"{'='*70}\n")

results = []
for i, bid in enumerate(bench_ids, 1):
    cls = BENCHMARKS[bid]
    bench = cls()
    print(f"[{i}/{len(bench_ids)}] {bench.name} ({bid}) — {bench.difficulty.value}")
    print(f"  Max turns: {bench.max_turns}")

    t0 = time.time()
    try:
        result = run_benchmark(bid)
        elapsed = time.time() - t0
        results.append(result)

        bar = lambda v: "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"  CAS:    {result.cas:.3f}  {bar(result.cas)}")
        print(f"  Fitness:{result.fitness:.3f}  Eff:{result.efficiency:.3f}  Em:{result.emergence:.3f}  Rob:{result.robustness:.3f}  Fi:{result.fidelity:.3f}")
        print(f"  Turns:  {len(result.turns)}  Time: {elapsed:.0f}s  Completed: {result.completed}")
        if result.error:
            print(f"  Error:  {result.error}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED: {e}  Time: {elapsed:.0f}s")
    print()

# Summary
if results:
    summary = aggregate_results(results)
    filepath = save_results(results, MODEL, "results")

    print(f"{'='*70}")
    print(f"  FINAL RESULTS — {MODEL}")
    print(f"{'='*70}")
    print()

    # Score table
    dims = summary["dimensions"]
    print(f"  {'Dimension':<15} {'Score':>8}  {'Bar'}")
    print(f"  {'─'*15} {'─'*8}  {'─'*20}")
    for dim in ["fitness", "efficiency", "emergence", "robustness", "fidelity"]:
        v = dims[dim]
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"  {dim.upper():<15} {v:>7.3f}  {bar}")
    print(f"  {'─'*15} {'─'*8}  {'─'*20}")
    print(f"  {'CAS':<15} {summary['overall_cas']:>7.3f}  {'█' * int(summary['overall_cas'] * 20)}{'░' * (20 - int(summary['overall_cas'] * 20))}")
    print()

    # Category table
    print(f"  Category Breakdown:")
    for cat, score in summary.get("by_category", {}).items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"    {cat:<15} {score:.3f}  {bar}")
    print()

    print(f"  Completed: {summary['completed']}/{summary['count']}")
    print(f"  Total time: {summary['total_latency_ms']/1000:.0f}s")
    print(f"  Results: {filepath}")
    print(f"{'='*70}")
