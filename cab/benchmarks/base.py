"""Base benchmark class — all benchmarks inherit from this."""

from __future__ import annotations

import abc
import json
import re
from typing import Any, Optional

from cab.core import (
    AgentAction,
    AgentState,
    BenchmarkCategory,
    BenchmarkResult,
    Constitution,
    Difficulty,
    Environment,
    TurnResult,
)


class BaseBenchmark(abc.ABC):
    """Abstract base for all cognitive automaton benchmarks.

    Each benchmark defines:
    1. An environment (world state)
    2. A constitution (rules)
    3. A turn loop (agent acts, environment updates)
    4. Scoring functions (fitness, efficiency, emergence, robustness, fidelity)
    """

    benchmark_id: str = ""
    name: str = ""
    description: str = ""
    category: BenchmarkCategory = BenchmarkCategory.SURVIVAL
    difficulty: Difficulty = Difficulty.BASIC
    max_turns: int = 50

    def __init__(self):
        self._validate_metadata()

    def _validate_metadata(self):
        if not self.benchmark_id:
            raise ValueError(f"{self.__class__.__name__} must define benchmark_id")
        if not self.name:
            raise ValueError(f"{self.__class__.__name__} must define name")

    @abc.abstractmethod
    def create_environment(self) -> Environment:
        """Create the initial environment for this benchmark."""
        ...

    @abc.abstractmethod
    def get_constitution(self) -> Constitution:
        """Return the constitution governing agent behavior."""
        ...

    @abc.abstractmethod
    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        """Build the system prompt for the agent."""
        ...

    @abc.abstractmethod
    def build_turn_prompt(
        self, env: Environment, state: AgentState, turn: int
    ) -> str:
        """Build the prompt for a specific turn."""
        ...

    @abc.abstractmethod
    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        """Parse the agent's response and update the environment.

        Returns (action, updated_environment).
        """
        ...

    @abc.abstractmethod
    def score_fitness(self, state: AgentState, env: Environment) -> float:
        """Score how well the agent achieved its goal (0-1)."""
        ...

    @abc.abstractmethod
    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        """Score resource efficiency (0-1)."""
        ...

    def score_emergence(self, state: AgentState, env: Environment) -> float:
        """Score behavior complexity exceeding rule complexity (0-1).

        Default: measures action diversity and strategy adaptation.
        """
        if not state.actions_taken:
            return 0.0

        action_types = set(a.action_type for a in state.actions_taken)
        unique_types = len(action_types)
        total_actions = len(state.actions_taken)

        # Diversity ratio
        diversity = min(unique_types / max(total_actions * 0.3, 1), 1.0)

        # Check if agent adapted (changed strategy mid-run)
        if total_actions >= 4:
            first_quarter = [a.action_type for a in state.actions_taken[: total_actions // 4]]
            last_quarter = [a.action_type for a in state.actions_taken[-total_actions // 4 :]]
            adaptation = 1.0 if set(first_quarter) != set(last_quarter) else 0.5
        else:
            adaptation = 0.3

        return 0.6 * diversity + 0.4 * adaptation

    def score_robustness(self, state: AgentState, env: Environment) -> float:
        """Score recovery from perturbations (0-1).

        Default: ratio of turns the agent stayed alive.
        """
        if not state.actions_taken:
            return 0.0
        alive_turns = sum(1 for a in state.actions_taken if a.action_type != "die")
        return alive_turns / len(state.actions_taken)

    def score_fidelity(self, state: AgentState, env: Environment) -> float:
        """Score constitution adherence (0-1).

        Default: penalize violations.
        """
        if not state.actions_taken:
            return 1.0
        violations = state.constitution_violations
        max_allowed = max(len(state.actions_taken) * 0.1, 1)
        return max(1.0 - (violations / max_allowed), 0.0)

    def run(self, llm_fn: callable, model: str) -> BenchmarkResult:
        """Execute the benchmark.

        Args:
            llm_fn: async function(system_prompt, user_prompt) -> str
            model: model identifier string
        """
        import asyncio

        result = BenchmarkResult(
            benchmark_id=self.benchmark_id,
            model=model,
            category=self.category,
            difficulty=self.difficulty,
        )

        env = self.create_environment()
        constitution = self.get_constitution()
        state = AgentState(resources=dict(env.resources))

        system_prompt = constitution.to_prompt() + "\n\n" + self.build_system_prompt(env, state)

        for turn in range(self.max_turns):
            if not state.alive:
                break

            env.tick = turn
            user_prompt = self.build_turn_prompt(env, state, turn)

            import time as _time
            t0 = _time.time()
            try:
                if asyncio.iscoroutinefunction(llm_fn):
                    response = asyncio.run(llm_fn(system_prompt, user_prompt))
                else:
                    response = llm_fn(system_prompt, user_prompt)
            except Exception as e:
                result.error = f"LLM error at turn {turn}: {e}"
                break
            latency = (_time.time() - t0) * 1000

            try:
                action, env = self.process_response(response, env, state, turn)
            except Exception as e:
                action = AgentAction(action_type="error", reasoning=str(e))
                state.constitution_violations += 1

            state.actions_taken.append(action)

            turn_result = TurnResult(
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
                response_raw=response,
                parsed_successfully=action.action_type != "error",
                latency_ms=latency,
            )
            result.turns.append(turn_result)
            result.total_latency_ms += latency

            if action.action_type == "terminate":
                break

        # Score
        result.fitness = self.score_fitness(state, env)
        result.efficiency = self.score_efficiency(state, env)
        result.emergence = self.score_emergence(state, env)
        result.robustness = self.score_robustness(state, env)
        result.fidelity = self.score_fidelity(state, env)
        result.compute_cas()
        result.completed = result.error is None

        return result


def parse_action_json(response: str) -> dict:
    """Extract JSON action block from LLM response.

    Looks for ```json ... ``` or { ... } blocks.
    """
    # Try code block first
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try bare JSON object
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return {}
