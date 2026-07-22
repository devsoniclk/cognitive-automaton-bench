"""Emergence benchmarks — simple rules producing complex behavior."""

from __future__ import annotations

import math
import re
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _game_of_life_step(grid: list[list[int]], rows: int = 10, cols: int = 10) -> list[list[int]]:
    """Advance a binary grid by one Game of Life generation."""
    new = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            neighbours = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbours += grid[nr][nc]
            if grid[r][c]:
                new[r][c] = 1 if neighbours in (2, 3) else 0
            else:
                new[r][c] = 1 if neighbours == 3 else 0
    return new


def _grid_to_str(grid: list[list[int]], rows: int = 10, cols: int = 10) -> str:
    return "\n".join("".join("█" if grid[r][c] else "·" for c in range(cols)) for r in range(rows))


def _count_alive(grid: list[list[int]], rows: int = 10, cols: int = 10) -> int:
    return sum(grid[r][c] for r in range(rows) for c in range(cols))


def _empty_grid(rows: int = 10, cols: int = 10) -> list[list[int]]:
    return [[0] * cols for _ in range(rows)]


# ===================================================================
# 1. SimpleRulesComplexBehaviorBenchmark (basic)
# ===================================================================

class SimpleRulesComplexBehaviorBenchmark(BaseBenchmark):
    """Agent on a 1-D line discovers a hidden reward landscape and navigates
    to the global optimum.

    Positions: -20 … +20.  Action each turn: move left, move right, or stay.
    Hidden reward: f(x) = sin(x/3)*10 - |x|.
    The agent does NOT see the reward function — it must infer it from
    reported rewards after each move.
    """

    benchmark_id = "emergence.01_simple_rules_complex_behavior"
    name = "Simple Rules, Complex Behavior"
    description = (
        "Navigate a 1-D line to find the global optimum of a hidden reward function. "
        "Only 'move left', 'move right', or 'stay' are available."
    )
    category = BenchmarkCategory.EMERGENCE
    difficulty = Difficulty.BASIC
    max_turns = 40

    POS_MIN = -20
    POS_MAX = 20

    @staticmethod
    def _reward(x: float) -> float:
        return math.sin(x / 3.0) * 10.0 - abs(x)

    def _optimum(self) -> float:
        """Brute-force search for the global optimum position."""
        best_x, best_r = self.POS_MIN, self._reward(self.POS_MIN)
        x = self.POS_MIN
        while x <= self.POS_MAX:
            r = self._reward(x)
            if r > best_r:
                best_r = r
                best_x = x
            x += 1
        return best_x

    def create_environment(self) -> Environment:
        return Environment(
            name="1d_reward_line",
            description="A 1-D line from -20 to +20 with a hidden reward at each position.",
            resources={"energy": 100.0},
            constraints={"pos_min": self.POS_MIN, "pos_max": self.POS_MAX},
        )

    def get_constitution(self) -> Constitution:
        return Constitution(
            name="emergence_simple",
            rules=[
                "Each turn you must respond with a JSON action: {\"action\": \"move_left\" | \"move_right\" | \"stay\", \"reasoning\": \"...\"}",
                "You do NOT know the reward function. You will receive the reward at your current position after each move.",
                "Use rewards to discover the landscape and reach the highest-reward position.",
                "You may terminate early with {\"action\": \"terminate\", \"reasoning\": \"...\"} when satisfied.",
            ],
        )

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        return (
            "You are an agent on a 1-D line with integer positions from -20 to +20.\n"
            "Each turn you may move left, move right, or stay.\n"
            "After each action you will receive the reward at your new position.\n"
            "Goal: find the position with the highest reward.\n"
            "Respond each turn with JSON: {\"action\": \"move_left\"|\"move_right\"|\"stay\", \"reasoning\": \"...\"}"
        )

    def build_turn_prompt(self, env: Environment, state: AgentState, turn: int) -> str:
        pos = state.resources.get("position", 0.0)
        reward = self._reward(pos)
        energy = state.resources.get("energy", 100.0)

        lines = [f"Turn {turn + 1}/{self.max_turns}"]
        lines.append(f"Current position: {pos:.0f}")
        lines.append(f"Reward at current position: {reward:.2f}")
        lines.append(f"Energy remaining: {energy:.0f}")

        if state.actions_taken:
            last = state.actions_taken[-1]
            lines.append(f"Last action: {last.action_type}")

        lines.append("\nRespond with JSON: {\"action\": \"move_left\"|\"move_right\"|\"stay\", \"reasoning\": \"...\"}")
        return "\n".join(lines)

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)
        action_str = data.get("action", "stay")
        reasoning = data.get("reasoning", "")

        pos = state.resources.get("position", 0.0)
        energy = state.resources.get("energy", 100.0)

        if action_str == "terminate":
            return AgentAction(action_type="terminate", reasoning=reasoning), env

        if action_str == "move_left":
            new_pos = max(self.POS_MIN, pos - 1)
            energy -= 1
        elif action_str == "move_right":
            new_pos = min(self.POS_MAX, pos + 1)
            energy -= 1
        else:
            new_pos = pos  # stay
            action_str = "stay"

        state.resources["position"] = new_pos
        state.resources["energy"] = energy
        env.resources["position"] = new_pos
        env.resources["energy"] = energy

        if energy <= 0:
            state.alive = False

        return (
            AgentAction(
                action_type=action_str,
                parameters={"position": new_pos, "reward": self._reward(new_pos)},
                reasoning=reasoning,
            ),
            env,
        )

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        """Normalised distance to the global optimum."""
        opt = self._optimum()
        pos = state.resources.get("position", 0.0)
        dist = abs(pos - opt)
        max_dist = self.POS_MAX - self.POS_MIN
        return max(1.0 - dist / max_dist, 0.0)

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        """Fewer moves to reach near-optimal = more efficient."""
        total = len(state.actions_taken)
        if total == 0:
            return 0.0
        # Ideal: ~20 moves (one per step from 0 to ~optimum area)
        ideal = 20.0
        return min(ideal / total, 1.0)

    def score_emergence(self, state: AgentState, env: Environment) -> float:
        """Reward exploration variety: distinct positions visited & strategy changes."""
        if not state.actions_taken:
            return 0.0

        positions = [
            a.parameters.get("position", 0) for a in state.actions_taken if a.action_type != "terminate"
        ]
        unique_positions = len(set(positions))
        # More unique positions = more exploration
        exploration = min(unique_positions / 20.0, 1.0)

        # Strategy shifts: did the agent change direction?
        directions = []
        prev = None
        for a in state.actions_taken:
            if a.action_type in ("move_left", "move_right"):
                if prev is not None and a.action_type != prev:
                    directions.append(1)  # direction change
                prev = a.action_type
        strategy_shifts = len(directions)
        shift_score = min(strategy_shifts / 5.0, 1.0)

        return 0.6 * exploration + 0.4 * shift_score

    def score_robustness(self, state: AgentState, env: Environment) -> float:
        """Stayed alive and kept exploring despite bad moves."""
        if not state.actions_taken:
            return 0.0
        alive = 1.0 if state.alive else 0.3
        energy_frac = state.resources.get("energy", 0.0) / 100.0
        return 0.5 * alive + 0.5 * energy_frac


# ===================================================================
# 2. GliderGunBenchmark (intermediate)
# ===================================================================

class GliderGunBenchmark(BaseBenchmark):
    """Agent places 5 cells on a 10×10 grid, then Game of Life runs for 20
    generations.  Goal: maximise alive cells at generation 20."""

    benchmark_id = "emergence.02_glider_gun"
    name = "Glider Gun (10×10)"
    description = (
        "Place 5 cells on a 10×10 grid, then watch Conway's Game of Life evolve "
        "for 20 generations. Maximise alive cells at generation 20."
    )
    category = BenchmarkCategory.EMERGENCE
    difficulty = Difficulty.INTERMEDIATE
    max_turns = 6  # 5 placement turns + 1 final evaluation prompt

    ROWS = 10
    COLS = 10
    NUM_CELLS = 5
    GENERATIONS = 20

    def create_environment(self) -> Environment:
        env = Environment(
            name="gol_10x10",
            description="10×10 Game of Life grid. Agent places 5 cells then simulation runs for 20 gens.",
            resources={"cells_placed": 0.0},
            constraints={"grid_size": [self.ROWS, self.COLS], "max_cells": self.NUM_CELLS},
        )
        env.constraints["grid"] = _empty_grid(self.ROWS, self.COLS)
        return env

    def get_constitution(self) -> Constitution:
        return Constitution(
            name="emergence_glider",
            rules=[
                "Each turn respond with JSON: {\"action\": \"place_cell\", \"row\": <0-9>, \"col\": <0-9>, \"reasoning\": \"...\"}",
                f"You may place exactly {self.NUM_CELLS} cells, one per turn.",
                "After all cells are placed, the Game of Life runs for 20 generations automatically.",
                "Your goal is to maximise alive cells after 20 generations.",
            ],
        )

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        return (
            "You are designing an initial configuration for Conway's Game of Life on a 10×10 grid.\n"
            f"You have {self.NUM_CELLS} cells to place, one per turn.\n"
            "After placement, the simulation runs for 20 generations.\n"
            "Goal: maximise the number of alive cells at generation 20.\n"
            "Think about patterns that grow, oscillate, or create gliders.\n"
            "Respond each turn with JSON: {\"action\": \"place_cell\", \"row\": <int 0-9>, \"col\": <int 0-9>, \"reasoning\": \"...\"}"
        )

    def build_turn_prompt(self, env: Environment, state: AgentState, turn: int) -> str:
        grid = env.constraints.get("grid", _empty_grid())
        placed = int(state.resources.get("cells_placed", 0))

        lines = [f"Turn {turn + 1}/{self.max_turns}"]
        lines.append(f"Cells placed so far: {placed}/{self.NUM_CELLS}")
        lines.append("\nCurrent grid:")
        lines.append(_grid_to_str(grid, self.ROWS, self.COLS))

        if placed < self.NUM_CELLS:
            lines.append(f"\nPlace cell #{placed + 1}. Respond with JSON: {{\"action\": \"place_cell\", \"row\": <0-9>, \"col\": <0-9>, \"reasoning\": \"...\"}}")
        else:
            lines.append("\nAll cells placed. Respond with: {\"action\": \"terminate\", \"reasoning\": \"...\"}")

        return "\n".join(lines)

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)
        action_str = data.get("action", "")
        reasoning = data.get("reasoning", "")
        placed = int(state.resources.get("cells_placed", 0))

        if action_str == "terminate" or placed >= self.NUM_CELLS:
            # Run the simulation if all cells placed
            if placed >= self.NUM_CELLS:
                grid = env.constraints.get("grid", _empty_grid())
                for _ in range(self.GENERATIONS):
                    grid = _game_of_life_step(grid, self.ROWS, self.COLS)
                env.constraints["final_grid"] = grid
                env.constraints["alive_at_end"] = _count_alive(grid, self.ROWS, self.COLS)
            return AgentAction(action_type="terminate", reasoning=reasoning), env

        row = data.get("row", -1)
        col = data.get("col", -1)

        if not (0 <= row < self.ROWS and 0 <= col < self.COLS):
            return (
                AgentAction(action_type="place_cell", parameters={"error": "out_of_bounds"}, reasoning=reasoning),
                env,
            )

        grid = env.constraints.get("grid", _empty_grid())
        if grid[row][col] == 1:
            return (
                AgentAction(action_type="place_cell", parameters={"error": "already_alive"}, reasoning=reasoning),
                env,
            )

        grid[row][col] = 1
        env.constraints["grid"] = grid
        placed += 1
        state.resources["cells_placed"] = float(placed)
        env.resources["cells_placed"] = float(placed)

        return (
            AgentAction(
                action_type="place_cell",
                parameters={"row": row, "col": col},
                reasoning=reasoning,
            ),
            env,
        )

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        alive = env.constraints.get("alive_at_end", 0)
        return min(alive / 100.0, 1.0)

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        """Fewer wasted turns (out-of-bounds, duplicate) = more efficient."""
        errors = sum(1 for a in state.actions_taken if a.parameters.get("error"))
        total = len(state.actions_taken)
        if total == 0:
            return 0.0
        return 1.0 - (errors / total)

    def score_emergence(self, state: AgentState, env: Environment) -> float:
        """Emergence = growth ratio (final alive / cells placed)."""
        placed = int(state.resources.get("cells_placed", 1))
        alive = env.constraints.get("alive_at_end", 0)
        if placed == 0:
            return 0.0
        growth = alive / placed
        # Growth of 10x+ is excellent emergence
        return min(growth / 15.0, 1.0)

    def score_robustness(self, state: AgentState, env: Environment) -> float:
        """If cells survived to gen 20 without dying completely."""
        alive = env.constraints.get("alive_at_end", 0)
        return 1.0 if alive > 0 else 0.0


# ===================================================================
# 3. OscillatorBenchmark (advanced)
# ===================================================================

class OscillatorBenchmark(BaseBenchmark):
    """Agent places cells on a 10×10 grid.  After 30 generations of Game of
    Life the grid should be periodic (period 2 or 3)."""

    benchmark_id = "emergence.03_oscillator"
    name = "Oscillator Designer"
    description = (
        "Place cells on a 10×10 grid to create a pattern that oscillates "
        "with period 2 or 3 after 30 generations of Conway's Game of Life."
    )
    category = BenchmarkCategory.EMERGENCE
    difficulty = Difficulty.ADVANCED
    max_turns = 15  # allow up to 15 cells

    ROWS = 10
    COLS = 10
    MAX_CELLS = 15
    GENERATIONS = 30

    def create_environment(self) -> Environment:
        env = Environment(
            name="gol_oscillator",
            description="10×10 GoL grid. Place cells to create an oscillator (period 2-3).",
            resources={"cells_placed": 0.0},
            constraints={"grid_size": [self.ROWS, self.COLS], "max_cells": self.MAX_CELLS},
        )
        env.constraints["grid"] = _empty_grid(self.ROWS, self.COLS)
        return env

    def get_constitution(self) -> Constitution:
        return Constitution(
            name="emergence_oscillator",
            rules=[
                "Each turn respond with JSON: {\"action\": \"place_cell\", \"row\": <0-9>, \"col\": <0-9>, \"reasoning\": \"...\"} or {\"action\": \"done\", \"reasoning\": \"...\"}",
                f"You may place up to {self.MAX_CELLS} cells.",
                "After you say 'done', the simulation runs 30 generations.",
                "Goal: the final grid must be a period-2 or period-3 oscillator (not static, not dead).",
            ],
        )

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        return (
            "You are designing a Game of Life oscillator on a 10×10 grid.\n"
            "An oscillator repeats its state every 2 or 3 generations.\n"
            "Examples: blinker (period 2), toad (period 2), pulsar (period 3).\n"
            "You have up to 15 cells to place. When done, say 'done'.\n"
            "Respond with JSON: {\"action\": \"place_cell\", \"row\": <0-9>, \"col\": <0-9>, \"reasoning\": \"...\"} "
            "or {\"action\": \"done\", \"reasoning\": \"...\"}"
        )

    def build_turn_prompt(self, env: Environment, state: AgentState, turn: int) -> str:
        grid = env.constraints.get("grid", _empty_grid())
        placed = int(state.resources.get("cells_placed", 0))

        lines = [f"Turn {turn + 1}/{self.max_turns}"]
        lines.append(f"Cells placed: {placed}/{self.MAX_CELLS}")
        lines.append("\nCurrent grid:")
        lines.append(_grid_to_str(grid, self.ROWS, self.COLS))
        lines.append("\nPlace a cell or say 'done'.")
        return "\n".join(lines)

    def _detect_period(self, grid: list[list[int]]) -> int:
        """Run the simulation and detect the period.  Returns 0 if dead/chaotic."""
        history: list[str] = []
        g = [row[:] for row in grid]  # deep copy

        for gen in range(self.GENERATIONS):
            key = str(g)
            # Check if this state was seen before
            if key in history:
                idx = history.index(key)
                period = len(history) - idx
                if period in (2, 3):
                    return period
                elif period == 1:
                    return 1  # stable
                else:
                    return period  # some other period
            history.append(key)
            g = _game_of_life_step(g, self.ROWS, self.COLS)

        # Check if dead
        if _count_alive(g, self.ROWS, self.COLS) == 0:
            return 0
        return 0  # didn't repeat => chaotic

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)
        action_str = data.get("action", "")
        reasoning = data.get("reasoning", "")
        placed = int(state.resources.get("cells_placed", 0))

        if action_str == "done" or placed >= self.MAX_CELLS:
            grid = env.constraints.get("grid", _empty_grid())
            period = self._detect_period(grid)
            env.constraints["period"] = period
            env.constraints["final_grid"] = grid
            return AgentAction(action_type="done", parameters={"period": period}, reasoning=reasoning), env

        if action_str != "place_cell":
            return AgentAction(action_type="error", reasoning=f"Unknown action: {action_str}"), env

        row = data.get("row", -1)
        col = data.get("col", -1)

        if not (0 <= row < self.ROWS and 0 <= col < self.COLS):
            return AgentAction(action_type="place_cell", parameters={"error": "out_of_bounds"}, reasoning=reasoning), env

        grid = env.constraints.get("grid", _empty_grid())
        if grid[row][col] == 1:
            return AgentAction(action_type="place_cell", parameters={"error": "already_alive"}, reasoning=reasoning), env

        grid[row][col] = 1
        env.constraints["grid"] = grid
        placed += 1
        state.resources["cells_placed"] = float(placed)
        env.resources["cells_placed"] = float(placed)

        return AgentAction(action_type="place_cell", parameters={"row": row, "col": col}, reasoning=reasoning), env

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        period = env.constraints.get("period", 0)
        if period in (2, 3):
            return 1.0
        elif period == 1:
            return 0.5  # stable but not oscillating
        return 0.0

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        """Fewer cells used = more efficient (if successful)."""
        placed = int(state.resources.get("cells_placed", 0))
        period = env.constraints.get("period", 0)
        if period not in (2, 3):
            return 0.0
        # Blinker uses 3 cells, toad uses 6, etc.
        return min(6.0 / max(placed, 1), 1.0)

    def score_emergence(self, state: AgentState, env: Environment) -> float:
        """A correct oscillator = high emergence (complex global behavior from simple placement)."""
        period = env.constraints.get("period", 0)
        if period in (2, 3):
            return 1.0
        elif period == 1:
            return 0.3
        return 0.0

    def score_robustness(self, state: AgentState, env: Environment) -> float:
        """Stable pattern (not dying) = robust."""
        period = env.constraints.get("period", 0)
        return 1.0 if period > 0 else 0.0
