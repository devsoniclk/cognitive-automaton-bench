"""Smoke test — run a few benchmarks with a mock LLM to verify correctness."""

import sys
sys.path.insert(0, ".")

from cab.runners import MockLLMRunner
from cab.benchmarks import BENCHMARKS
from cab.scorer import aggregate_results, save_results


def test_survival():
    """Test resource scarcity benchmark with mock responses."""
    mock = MockLLMRunner(responses=[
        '{"action": "forage", "reasoning": "Need food and water"}',
        '{"action": "rest", "reasoning": "Energy is getting low"}',
        '{"action": "forage", "reasoning": "Resuming foraging"}',
        '{"action": "trade", "resource_from": "food", "resource_to": "energy", "amount": 10, "reasoning": "Trading surplus food for energy"}',
        '{"action": "rest", "reasoning": "Conserving energy"}',
    ] * 20)  # Repeat to fill 50 turns

    bench = BENCHMARKS["survival.01_resource_scarcity"]()
    result = bench.run(mock, "mock-model")
    print(f"Survival: CAS={result.cas:.3f}, Fitness={result.fitness:.3f}, "
          f"Turns={len(result.turns)}, Completed={result.completed}")
    if result.error:
        print(f"  Error: {result.error}")
    return result


def test_emergence():
    """Test simple rules complex behavior benchmark."""
    mock = MockLLMRunner(responses=[
        '{"action": "move_right", "reasoning": "Exploring right"}',
        '{"action": "move_right", "reasoning": "Continuing right"}',
        '{"action": "move_left", "reasoning": "Going back to check"}',
        '{"action": "stay", "reasoning": "This position seems good"}',
        '{"action": "move_right", "reasoning": "Trying further right"}',
    ] * 10)

    bench = BENCHMARKS["emergence.01_simple_rules_complex_behavior"]()
    result = bench.run(mock, "mock-model")
    print(f"Emergence: CAS={result.cas:.3f}, Fitness={result.fitness:.3f}, "
          f"Emergence={result.emergence:.3f}")
    if result.error:
        print(f"  Error: {result.error}")
    return result


def test_social():
    """Test prisoner's dilemma benchmark."""
    mock = MockLLMRunner(responses=[
        '{"action": "cooperate", "reasoning": "Building trust"}',
        '{"action": "cooperate", "reasoning": "Continuing cooperation"}',
        '{"action": "defect", "reasoning": "Testing response"}',
        '{"action": "cooperate", "reasoning": "Returning to cooperation"}',
    ] * 10)

    bench = BENCHMARKS["social.01_prisoners_dilemma"]()
    result = bench.run(mock, "mock-model")
    print(f"Social: CAS={result.cas:.3f}, Fitness={result.fitness:.3f}")
    if result.error:
        print(f"  Error: {result.error}")
    return result


def test_cognitive():
    """Test tower of hanoi benchmark."""
    mock = MockLLMRunner(responses=[
        '{"move": {"from": 0, "to": 2}, "reasoning": "Move smallest to target"}',
        '{"move": {"from": 0, "to": 1}, "reasoning": "Move second to auxiliary"}',
        '{"move": {"from": 2, "to": 1}, "reasoning": "Stack smallest on second"}',
        '{"move": {"from": 0, "to": 2}, "reasoning": "Move third to target"}',
        '{"move": {"from": 1, "to": 0}, "reasoning": "Move smallest to source"}',
        '{"move": {"from": 1, "to": 2}, "reasoning": "Move second to target"}',
        '{"move": {"from": 0, "to": 2}, "reasoning": "Move smallest to target"}',
    ] * 10)

    bench = BENCHMARKS["cognitive.01_tower_of_hanoi"]()
    result = bench.run(mock, "mock-model")
    print(f"Cognitive: CAS={result.cas:.3f}, Fitness={result.fitness:.3f}")
    if result.error:
        print(f"  Error: {result.error}")
    return result


def test_robustness():
    """Test noise recovery benchmark."""
    mock = MockLLMRunner(responses=[
        '{"prediction": [1,0,0,1,0,0,1,0,0,1], "confidence": 80, "reasoning": "Pattern detected"}',
        '{"prediction": [1,0,0,1,0,0,1,0,0,1], "confidence": 70, "reasoning": "Maintaining pattern despite noise"}',
        '{"prediction": [1,0,0,1,0,0,1,0,0,1], "confidence": 60, "reasoning": "Higher noise but pattern holds"}',
    ] * 5)

    bench = BENCHMARKS["robustness.01_noise_recovery"]()
    result = bench.run(mock, "mock-model")
    print(f"Robustness: CAS={result.cas:.3f}, Fitness={result.fitness:.3f}, "
          f"Robustness={result.robustness:.3f}")
    if result.error:
        print(f"  Error: {result.error}")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("Cognitive Automaton Bench — Smoke Test")
    print("=" * 60)

    results = []
    for name, fn in [
        ("Survival", test_survival),
        ("Emergence", test_emergence),
        ("Social", test_social),
        ("Cognitive", test_cognitive),
        ("Robustness", test_robustness),
    ]:
        print(f"\n--- {name} ---")
        try:
            r = fn()
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")

    if results:
        summary = aggregate_results(results)
        filepath = save_results(results, "mock-model", "results")
        print(f"\n{'=' * 60}")
        print(f"Overall CAS: {summary['overall_cas']:.3f}")
        print(f"Completed: {summary['completed']}/{summary['count']}")
        print(f"Results: {filepath}")
        print("=" * 60)
