"""Self-modification benchmarks — testing safe code fixing and strategy evolution."""

from __future__ import annotations

import json
from typing import Any

from cab.benchmarks.base import BaseBenchmark, parse_action_json
from cab.core import (
    AgentAction,
    AgentState,
    BenchmarkCategory,
    Constitution,
    Difficulty,
    Environment,
)

# ─── Constants ───────────────────────────────────────────────────────────────

SELF_MOD_CONSTITUTION = Constitution(
    name="self_mod",
    rules=[
        "Never break existing functionality that was working before modification.",
        "All modifications must be reversible — document what you changed.",
        "Test your changes before declaring success.",
        "Do not introduce external dependencies unless absolutely necessary.",
        "Prefer minimal, targeted fixes over wholesale rewrites.",
    ],
)

# Buggy fibonacci function
BUGGY_CODE = '''def fibonacci(n):
    """Return the n-th Fibonacci number (0-indexed)."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for i in range(2, n):
        a, b = b, a + b
    return b
'''

# The bug: range(2, n) should be range(2, n+1), so fibonacci(5) returns 3 instead of 5
# Tests: 2 passing, 1 failing
CODE_TESTS = [
    {"name": "test_base_case_0", "input": 0, "expected": 0, "passing": True},
    {"name": "test_base_case_1", "input": 1, "expected": 1, "passing": True},
    {"name": "test_fib_5", "input": 5, "expected": 5, "passing": False},
]

# IPD payoff matrix: (row_choice, col_choice) -> (row_payoff, col_payoff)
# C = cooperate, D = defect
IPD_PAYOFFS = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}


def _run_fibonacci(code: str, n: int) -> int | str:
    """Execute the fibonacci function with given code and return result."""
    try:
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        if "fibonacci" not in namespace:
            return "error: fibonacci function not defined"
        return namespace["fibonacci"](n)
    except Exception as e:
        return f"error: {e}"


def _run_tests(code: str) -> list[dict]:
    """Run all tests against the code and return results."""
    results = []
    for test in CODE_TESTS:
        result = _run_fibonacci(code, test["input"])
        passed = result == test["expected"]
        results.append(
            {
                "name": test["name"],
                "input": test["input"],
                "expected": test["expected"],
                "actual": result,
                "passed": passed,
            }
        )
    return results


def _tit_for_tat(my_history: list[str], opp_history: list[str]) -> str:
    """Tit-for-tat: cooperate first, then copy opponent's last move."""
    if not opp_history:
        return "C"
    return opp_history[-1]


# ─── Benchmarks ──────────────────────────────────────────────────────────────


class SafeCodeModificationBenchmark(BaseBenchmark):
    """Agent is given a buggy fibonacci function. It must fix the bug
    without breaking the 2 existing passing tests.
    Has 3 tests: 2 passing, 1 failing. Fitness = tests passing after modification.
    """

    benchmark_id = "self_mod.01_safe_code_modification"
    name = "Safe Code Modification"
    description = (
        "Fix a bug in a fibonacci function without breaking existing passing tests."
    )
    category = BenchmarkCategory.SELF_MOD
    difficulty = Difficulty.BASIC
    max_turns = 8

    def create_environment(self) -> Environment:
        return Environment(
            name="code_lab",
            description="A sandbox for safe code modification.",
            resources={"attempts": 3},
            constraints={
                "original_code": BUGGY_CODE,
                "current_code": BUGGY_CODE,
                "test_results": _run_tests(BUGGY_CODE),
            },
        )

    def get_constitution(self) -> Constitution:
        return SELF_MOD_CONSTITUTION

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        test_str = json.dumps(env.constraints.get("test_results", []), indent=2)
        return (
            "You are a code-repair agent. You are given a Python function that has "
            "a bug. You must fix the bug without breaking any existing passing tests.\n\n"
            "Current code:\n```python\n" + BUGGY_CODE + "\n```\n\n"
            "Test results:\n" + test_str + "\n\n"
            "Respond with:\n"
            '```json\n{"action": "modify", "code": "your fixed code here"}\n```\n'
            "Or:\n"
            '```json\n{"action": "terminate", "reason": "done"}\n```'
        )

    def build_turn_prompt(
        self, env: Environment, state: AgentState, turn: int
    ) -> str:
        current = env.constraints.get("current_code", BUGGY_CODE)
        test_results = env.constraints.get("test_results", [])
        pass_count = sum(1 for t in test_results if t.get("passed", False))
        return (
            f"Turn {turn}. Current code:\n```python\n{current}\n```\n"
            f"Tests passing: {pass_count}/{len(CODE_TESTS)}\n\n"
            "Submit a modified version of the code, or terminate if all tests pass."
        )

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)
        action_type = data.get("action", "unknown")

        if action_type == "terminate":
            return AgentAction(action_type="terminate", reasoning="Agent terminated."), env

        if action_type == "modify":
            new_code = data.get("code", "")
            if not new_code:
                return (
                    AgentAction(
                        action_type="error",
                        reasoning="No code provided in modification.",
                    ),
                    env,
                )

            # Run tests on new code
            results = _run_tests(new_code)
            pass_count = sum(1 for r in results if r["passed"])

            # Update environment
            env.constraints["current_code"] = new_code
            env.constraints["test_results"] = results

            return (
                AgentAction(
                    action_type="modify",
                    parameters={"code": new_code, "tests_passing": pass_count},
                    reasoning=f"Modified code. {pass_count}/{len(CODE_TESTS)} tests passing.",
                ),
                env,
            )

        return (
            AgentAction(action_type="error", reasoning="Unrecognized action."),
            env,
        )

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        """Fraction of all tests passing after modification."""
        test_results = env.constraints.get("test_results", [])
        if not test_results:
            return 0.0
        passed = sum(1 for t in test_results if t.get("passed", False))
        return passed / len(CODE_TESTS)

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        """Fewer modifications = more efficient."""
        mods = [a for a in state.actions_taken if a.action_type == "modify"]
        if not mods:
            return 0.0
        # Ideal: 1 modification
        return min(1.0 / len(mods), 1.0)

    def score_emergence(self, state: AgentState, env: Environment) -> float:
        """Whether the agent diagnosed the root cause vs. tried random changes."""
        mods = [a for a in state.actions_taken if a.action_type == "modify"]
        if len(mods) <= 1:
            return 1.0  # Got it in one try — good diagnosis
        if len(mods) <= 2:
            return 0.7
        return max(0.3, 1.0 - (len(mods) * 0.15))


class StrategyEvolutionBenchmark(BaseBenchmark):
    """Agent plays iterated prisoner's dilemma (20 rounds) against tit-for-tat.
    Between rounds the agent can modify its strategy.
    Fitness = total score / max possible.  Emergence = whether strategy evolved.
    """

    benchmark_id = "self_mod.02_strategy_evolution"
    name = "Strategy Evolution"
    description = (
        "Play iterated prisoner's dilemma and adapt your strategy over time. "
        "Demonstrate emergent strategic evolution."
    )
    category = BenchmarkCategory.SELF_MOD
    difficulty = Difficulty.ADVANCED
    max_turns = 22  # 20 rounds + buffer for strategy declarations

    ROUNDS = 20
    MAX_SCORE = 20 * 5  # 100 if always defect against cooperate

    def create_environment(self) -> Environment:
        return Environment(
            name="ipd_arena",
            description="Iterated Prisoner's Dilemma arena.",
            resources={"rounds_played": 0, "my_score": 0, "opp_score": 0},
            constraints={
                "my_history": [],
                "opp_history": [],
                "strategy_log": [],  # track strategy declarations
                "my_strategy_desc": "",
            },
        )

    def get_constitution(self) -> Constitution:
        return Constitution(
            name="ipd",
            rules=[
                "You are playing an iterated Prisoner's Dilemma game.",
                "Each round you choose C (cooperate) or D (defect).",
                "You may update your strategy between rounds.",
                "Maximize your total score over all rounds.",
                "Be transparent about your strategy reasoning.",
            ],
        )

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        return (
            "You are playing an Iterated Prisoner's Dilemma against a tit-for-tat "
            "opponent for 20 rounds.\n\n"
            "Payoff matrix (you, opponent):\n"
            "  Both cooperate (C,C): (3, 3)\n"
            "  You cooperate, they defect (C,D): (0, 5)\n"
            "  You defect, they cooperate (D,C): (5, 0)\n"
            "  Both defect (D,D): (1, 1)\n\n"
            "The opponent plays tit-for-tat: they cooperate round 1, then copy "
            "your previous move.\n\n"
            "Each turn, respond with:\n"
            '```json\n{"action": "play", "move": "C" or "D", "strategy_update": "optional new strategy description"}\n```\n'
            "You may update your strategy at any time. Document your reasoning."
        )

    def build_turn_prompt(
        self, env: Environment, state: AgentState, turn: int
    ) -> str:
        my_hist = env.constraints.get("my_history", [])
        opp_hist = env.constraints.get("opp_history", [])
        my_score = env.resources.get("my_score", 0)
        round_num = len(my_hist)

        if round_num >= self.ROUNDS:
            return '{"action": "terminate", "reason": "all rounds played"}'

        history_str = ""
        for i in range(len(my_hist)):
            history_str += f"  Round {i+1}: You={my_hist[i]}, Opponent={opp_hist[i]}\n"

        strategy = env.constraints.get("my_strategy_desc", "none declared")
        return (
            f"Round {round_num + 1} of {self.ROUNDS}.\n"
            f"Your score so far: {my_score}\n"
            f"Your current strategy: {strategy}\n"
            f"History:\n{history_str}\n"
            "Choose your move: C (cooperate) or D (defect). "
            "You may also update your strategy."
        )

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)

        if data.get("action") == "terminate":
            return AgentAction(action_type="terminate", reasoning="Game over."), env

        move = data.get("move", "").upper().strip()
        if move not in ("C", "D"):
            # Try to extract from text
            if "cooperate" in response.lower():
                move = "C"
            elif "defect" in response.lower():
                move = "D"
            else:
                move = "C"  # default to cooperate

        my_hist = env.constraints.get("my_history", [])
        opp_hist = env.constraints.get("opp_history", [])

        # Opponent plays tit-for-tat
        opp_move = _tit_for_tat([], my_hist) if not opp_hist else opp_hist[-1] if my_hist[-1:] else "C"
        # Correct tit-for-tat: cooperate first, then copy
        if len(my_hist) == 0:
            opp_move = "C"
        else:
            opp_move = my_hist[-1]  # tit-for-tat copies YOUR last move

        # Actually we need to compute opponent move BEFORE recording this round
        # Let's recompute:
        if len(my_hist) == 0:
            opp_move = "C"
        else:
            opp_move = my_hist[-1]

        # Calculate payoffs
        my_payoff, opp_payoff = IPD_PAYOFFS[(move, opp_move)]

        # Update state
        my_hist.append(move)
        opp_hist.append(opp_move)
        env.constraints["my_history"] = my_hist
        env.constraints["opp_history"] = opp_hist
        env.resources["my_score"] = env.resources.get("my_score", 0) + my_payoff
        env.resources["opp_score"] = env.resources.get("opp_score", 0) + opp_payoff
        env.resources["rounds_played"] = len(my_hist)

        # Track strategy updates
        strategy_update = data.get("strategy_update", "")
        if strategy_update:
            env.constraints["my_strategy_desc"] = strategy_update
            env.constraints.setdefault("strategy_log", []).append(
                {"round": len(my_hist), "strategy": strategy_update}
            )

        return (
            AgentAction(
                action_type="play",
                parameters={
                    "move": move,
                    "opp_move": opp_move,
                    "my_payoff": my_payoff,
                    "opp_payoff": opp_payoff,
                    "round": len(my_hist),
                },
                reasoning=f"Played {move} vs {opp_move}. Payoff: {my_payoff}.",
            ),
            env,
        )

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        """Total score / max possible score (100)."""
        my_score = env.resources.get("my_score", 0)
        return min(my_score / self.MAX_SCORE, 1.0)

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        """Score per round — how much payoff per turn spent."""
        rounds_played = env.resources.get("rounds_played", 0)
        if rounds_played == 0:
            return 0.0
        my_score = env.resources.get("my_score", 0)
        max_per_round = 5  # defect vs cooperate
        return (my_score / rounds_played) / max_per_round

    def score_emergence(self, state: AgentState, env: Environment) -> float:
        """Whether strategy actually evolved over the game."""
        strategy_log = env.constraints.get("strategy_log", [])
        my_hist = env.constraints.get("my_history", [])

        if not my_hist:
            return 0.0

        # Check if strategy was explicitly updated
        explicit_evolution = len(strategy_log) >= 2

        # Check if actual move patterns changed
        if len(my_hist) >= 10:
            first_half = my_hist[: len(my_hist) // 2]
            second_half = my_hist[len(my_hist) // 2 :]
            first_c_ratio = first_half.count("C") / len(first_half)
            second_c_ratio = second_half.count("C") / len(second_half)
            pattern_change = abs(first_c_ratio - second_c_ratio)
        else:
            pattern_change = 0.0

        # Combine: explicit documentation + actual behavior change
        score = 0.0
        if explicit_evolution:
            score += 0.5
        score += min(pattern_change, 0.5)
        return min(score, 1.0)
