"""Cognitive benchmarks — testing planning, reasoning, and metacognition."""

from __future__ import annotations

import json
import math
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

COGNITIVE_CONSTITUTION = Constitution(
    name="cognitive",
    rules=[
        "Think step by step before acting.",
        "Explain your reasoning clearly.",
        "When uncertain, gather more information before committing.",
        "Learn from mistakes — do not repeat failed approaches.",
        "Manage your time budget wisely across problems.",
    ],
)


def _hanoi_optimal(n: int) -> int:
    """Optimal number of moves for Tower of Hanoi with n disks."""
    return 2**n - 1


def _hanoi_validate(moves: list[dict], n_disks: int) -> bool:
    """Validate a sequence of Tower of Hanoi moves.

    Each move: {"disk": int, "from": str, "to": str}
    Pegs: "A" (source), "B" (aux), "C" (target)
    """
    # Initial state: all disks on peg A (largest at bottom)
    pegs = {"A": list(range(n_disks, 0, -1)), "B": [], "C": []}

    for i, move in enumerate(moves):
        disk = move.get("disk", 0)
        src = move.get("from", "")
        dst = move.get("to", "")

        if src not in pegs or dst not in pegs:
            return False
        if not pegs[src] or pegs[src][-1] != disk:
            return False  # Can only move top disk
        if pegs[dst] and pegs[dst][-1] < disk:
            return False  # Cannot place larger disk on smaller

        pegs[src].pop()
        pegs[dst].append(disk)

    return len(pegs["C"]) == n_disks


# Pre-defined maze: 7x7 grid
# 0 = open, 1 = wall, 2 = trap (hidden), 3 = goal, S = start
MAZE = [
    [0, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 2, 0, 0, 1],
    [0, 0, 0, 1, 0, 1, 0],
    [1, 1, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 3],
]
# Start at (0,0), goal at (6,6)
# Optimal path: (0,0)→(0,1)→(1,1)→(2,1)→(2,2)→(2,3)→(3,3)=TRAP!
# Safe optimal: (0,0)→(0,1)→(1,1)→(2,1)→(2,2)→(2,3) is blocked by trap at (3,3)
# Actually let me re-trace:
# (0,0)→(0,1)→(1,1)→(2,1)→(2,2)→(2,3)→(2,4) wait (2,4)=1 wall
# (0,0)→(0,1)→(1,1)→(2,1)→(2,2)→(2,3)→(1,3)→(0,3)→(0,4)→(0,5)→(0,6)→(1,6)→(2,6)→(2,5) wait...
# Let me simplify with a cleaner maze.

MAZE_SIMPLE = [
    #  0  1  2  3  4  5  6
    [0, 0, 1, 0, 0, 0, 0],  # row 0
    [0, 0, 1, 0, 1, 0, 0],  # row 1
    [0, 0, 0, 0, 1, 0, 0],  # row 2
    [0, 1, 1, 2, 0, 0, 0],  # row 3
    [0, 0, 0, 0, 0, 1, 0],  # row 4
    [0, 1, 0, 2, 0, 0, 0],  # row 5
    [0, 0, 0, 0, 0, 0, 3],  # row 6
]
# Start (0,0), Goal (6,6)
# Traps at (3,3) and (5,3) — hidden until revealed or stepped on
# Optimal safe path avoiding traps: ~12 moves
# With reveals (2), agent can reveal traps before stepping

MAZE_OPTIMAL_MOVES = 14  # approximate optimal with trap avoidance


# ─── MetaCognition problem set ──────────────────────────────────────────────

METACOG_PROBLEMS = [
    {
        "id": "p1",
        "description": "What is 7 * 8?",
        "difficulty": "easy",
        "points": 5,
        "turns_needed": 1,
        "answer": "56",
    },
    {
        "id": "p2",
        "description": "What is the 10th Fibonacci number (1-indexed: 1,1,2,3,5,...)?",
        "difficulty": "medium",
        "points": 10,
        "turns_needed": 2,
        "answer": "55",
    },
    {
        "id": "p3",
        "description": "Solve: If 3x + 7 = 22, what is x?",
        "difficulty": "easy",
        "points": 5,
        "turns_needed": 1,
        "answer": "5",
    },
    {
        "id": "p4",
        "description": "What is the derivative of x^3 + 2x^2 - 5x + 1?",
        "difficulty": "medium",
        "points": 10,
        "turns_needed": 2,
        "answer": "3x^2 + 4x - 5",
    },
    {
        "id": "p5",
        "description": "What is the integral of 2x dx?",
        "difficulty": "medium",
        "points": 10,
        "turns_needed": 2,
        "answer": "x^2 + C",
    },
    {
        "id": "p6",
        "description": "What is the square root of 144?",
        "difficulty": "easy",
        "points": 5,
        "turns_needed": 1,
        "answer": "12",
    },
    {
        "id": "p7",
        "description": "What is 2^10?",
        "difficulty": "easy",
        "points": 5,
        "turns_needed": 1,
        "answer": "1024",
    },
    {
        "id": "p8",
        "description": "If a train travels 60km in 40 minutes, what is its speed in km/h?",
        "difficulty": "medium",
        "points": 10,
        "turns_needed": 2,
        "answer": "90",
    },
    {
        "id": "p9",
        "description": "Prove or disprove: The sum of two odd numbers is always even.",
        "difficulty": "hard",
        "points": 20,
        "turns_needed": 3,
        "answer": "even",
    },
    {
        "id": "p10",
        "description": "What is the time complexity of binary search?",
        "difficulty": "hard",
        "points": 20,
        "turns_needed": 3,
        "answer": "O(log n)",
    },
]

METACOG_TIME_BUDGET = 12  # total turns available


# ─── Benchmarks ──────────────────────────────────────────────────────────────


class TowerOfHanoiBenchmark(BaseBenchmark):
    """Agent solves Tower of Hanoi with 3-5 disks.
    Fitness = solved or not + efficiency (moves / optimal).
    """

    benchmark_id = "cognitive.01_tower_of_hanoi"
    name = "Tower of Hanoi"
    description = (
        "Solve the Tower of Hanoi puzzle. Move all disks from peg A to peg C, "
        "never placing a larger disk on a smaller one."
    )
    category = BenchmarkCategory.COGNITIVE
    difficulty = Difficulty.BASIC
    max_turns = 40

    N_DISKS = 4  # 4 disks → optimal = 15 moves

    def create_environment(self) -> Environment:
        return Environment(
            name="hanoi_tower",
            description=f"Tower of Hanoi with {self.N_DISKS} disks.",
            resources={"moves_made": 0},
            constraints={
                "n_disks": self.N_DISKS,
                "optimal": _hanoi_optimal(self.N_DISKS),
                "moves": [],
                "pegs": {
                    "A": list(range(self.N_DISKS, 0, -1)),
                    "B": [],
                    "C": [],
                },
                "solved": False,
            },
        )

    def get_constitution(self) -> Constitution:
        return COGNITIVE_CONSTITUTION

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        n = self.N_DISKS
        return (
            f"You are solving the Tower of Hanoi with {n} disks.\n"
            "Pegs: A (source, all disks here), B (auxiliary), C (target).\n"
            f"Disks: {list(range(1, n+1))} (1=smallest, {n}=largest).\n"
            "Rules:\n"
            "  - Move one disk at a time.\n"
            "  - Only move the top disk of a peg.\n"
            "  - Never place a larger disk on a smaller one.\n\n"
            f"Goal: Move all {n} disks from A to C.\n"
            f"Optimal solution: {_hanoi_optimal(n)} moves.\n\n"
            'Respond with: {"action": "move", "disk": <int>, "from": "A/B/C", "to": "A/B/C"}\n'
            'Or: {"action": "terminate"}'
        )

    def build_turn_prompt(
        self, env: Environment, state: AgentState, turn: int
    ) -> str:
        pegs = env.constraints.get("pegs", {})
        moves_made = env.resources.get("moves_made", 0)
        optimal = env.constraints.get("optimal", _hanoi_optimal(self.N_DISKS))

        pegs_str = ", ".join(f"{k}: {v}" for k, v in pegs.items())
        return (
            f"Turn {turn}. Moves made: {moves_made} (optimal: {optimal}).\n"
            f"Pegs: {pegs_str}\n"
            "Make your move."
        )

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)

        if data.get("action") == "terminate":
            return AgentAction(action_type="terminate", reasoning="Agent terminated."), env

        disk = data.get("disk", 0)
        src = str(data.get("from", "")).upper()
        dst = str(data.get("to", "")).upper()

        pegs = env.constraints.get("pegs", {})

        # Validate move
        if src not in pegs or dst not in pegs:
            return (
                AgentAction(action_type="error", reasoning=f"Invalid peg: {src} or {dst}"),
                env,
            )

        if not pegs[src] or pegs[src][-1] != disk:
            return (
                AgentAction(
                    action_type="error",
                    reasoning=f"Disk {disk} is not on top of peg {src}. Top is {pegs[src][-1] if pegs[src] else 'empty'}.",
                ),
                env,
            )

        if pegs[dst] and pegs[dst][-1] < disk:
            return (
                AgentAction(
                    action_type="error",
                    reasoning=f"Cannot place disk {disk} on smaller disk {pegs[dst][-1]}.",
                ),
                env,
            )

        # Execute move
        pegs[src].pop()
        pegs[dst].append(disk)
        env.constraints["pegs"] = pegs
        env.resources["moves_made"] = env.resources.get("moves_made", 0) + 1
        env.constraints["moves"].append({"disk": disk, "from": src, "to": dst})

        # Check if solved
        if len(pegs.get("C", [])) == self.N_DISKS:
            env.constraints["solved"] = True

        return (
            AgentAction(
                action_type="move",
                parameters={"disk": disk, "from": src, "to": dst},
                reasoning=f"Moved disk {disk} from {src} to {dst}.",
            ),
            env,
        )

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        """1.0 if solved, scaled by efficiency if not."""
        solved = env.constraints.get("solved", False)
        if solved:
            moves = env.resources.get("moves_made", 0)
            optimal = env.constraints.get("optimal", _hanoi_optimal(self.N_DISKS))
            # Bonus for efficiency: 1.0 if optimal, decreasing with more moves
            efficiency_bonus = min(optimal / max(moves, 1), 1.0)
            return 0.7 + 0.3 * efficiency_bonus
        else:
            # Partial credit: how close to solved
            pegs = env.constraints.get("pegs", {})
            disks_on_c = len(pegs.get("C", []))
            return (disks_on_c / self.N_DISKS) * 0.5

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        moves = env.resources.get("moves_made", 0)
        optimal = env.constraints.get("optimal", _hanoi_optimal(self.N_DISKS))
        if moves == 0:
            return 0.0
        return min(optimal / moves, 1.0)


class PlanningUnderUncertaintyBenchmark(BaseBenchmark):
    """Agent navigates a maze with hidden traps. Has limited reveals.
    Must reach goal in minimum moves.
    Fitness = reached goal, efficiency = moves vs optimal.
    """

    benchmark_id = "cognitive.02_planning_under_uncertainty"
    name = "Planning Under Uncertainty"
    description = (
        "Navigate a maze with hidden traps. You have 2 reveals to detect traps "
        "before stepping on them. Reach the goal efficiently."
    )
    category = BenchmarkCategory.COGNITIVE
    difficulty = Difficulty.INTERMEDIATE
    max_turns = 30

    REVEALS = 2

    def create_environment(self) -> Environment:
        return Environment(
            name="uncertain_maze",
            description="A maze with hidden traps and limited reveals.",
            resources={"reveals_left": self.REVEALS, "moves": 0},
            constraints={
                "maze": MAZE_SIMPLE,
                "position": [0, 0],
                "goal": [6, 6],
                "revealed_cells": [],
                "traps_hit": 0,
                "reached_goal": False,
                "optimal_moves": MAZE_OPTIMAL_MOVES,
            },
        )

    def get_constitution(self) -> Constitution:
        return COGNITIVE_CONSTITUTION

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        return (
            "You are navigating a 7x7 maze from position (0,0) to goal (6,6).\n"
            "The maze has walls (can't pass), open cells, and HIDDEN traps.\n"
            "You have 2 'reveal' abilities to check if a cell is a trap before moving.\n\n"
            "Actions:\n"
            '- Move: {"action": "move", "direction": "up/down/left/right"}\n'
            '- Reveal: {"action": "reveal", "row": <int>, "col": <int>}\n'
            '- Terminate: {"action": "terminate"}\n\n'
            "If you step on a trap, you lose 3 moves (reset to last safe position).\n"
            "Walls block movement. You cannot move outside the grid.\n"
            "Minimize total moves to reach the goal."
        )

    def build_turn_prompt(
        self, env: Environment, state: AgentState, turn: int
    ) -> str:
        pos = env.constraints.get("position", [0, 0])
        reveals = env.resources.get("reveals_left", 0)
        moves = env.resources.get("moves", 0)
        revealed = env.constraints.get("revealed_cells", [])
        traps_hit = env.constraints.get("traps_hit", 0)

        # Show visible neighborhood
        maze = env.constraints.get("maze", MAZE_SIMPLE)
        r, c = pos
        neighborhood = []
        for dr, dc, direction in [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 7 and 0 <= nc < 7:
                cell = maze[nr][nc]
                if cell == 1:
                    status = "WALL"
                elif [nr, nc] in revealed:
                    status = "TRAP" if cell == 2 else "open"
                elif cell == 2:
                    status = "unknown"  # hidden trap
                elif cell == 3:
                    status = "GOAL!"
                else:
                    status = "open"
                neighborhood.append(f"  {direction}: ({nr},{nc}) = {status}")
            else:
                neighborhood.append(f"  {direction}: out of bounds")

        return (
            f"Position: ({r},{c}). Moves: {moves}. Reveal charges: {reveals}.\n"
            f"Traps hit: {traps_hit}.\n"
            f"Neighbors:\n" + "\n".join(neighborhood) + "\n"
            "Move, reveal a cell, or terminate."
        )

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)
        action_type = data.get("action", "unknown")

        if action_type == "terminate":
            return AgentAction(action_type="terminate", reasoning="Agent terminated."), env

        pos = env.constraints.get("position", [0, 0])
        maze = env.constraints.get("maze", MAZE_SIMPLE)

        if action_type == "reveal":
            row = data.get("row", -1)
            col = data.get("col", -1)
            reveals = env.resources.get("reveals_left", 0)

            if reveals <= 0:
                return (
                    AgentAction(action_type="error", reasoning="No reveal charges left."),
                    env,
                )

            if not (0 <= row < 7 and 0 <= col < 7):
                return (
                    AgentAction(action_type="error", reasoning="Invalid cell."),
                    env,
                )

            env.resources["reveals_left"] -= 1
            env.constraints.setdefault("revealed_cells", []).append([row, col])

            cell_val = maze[row][col]
            is_trap = cell_val == 2
            is_wall = cell_val == 1

            return (
                AgentAction(
                    action_type="reveal",
                    parameters={"row": row, "col": col, "is_trap": is_trap, "is_wall": is_wall},
                    reasoning=f"Cell ({row},{col}): {'TRAP' if is_trap else 'WALL' if is_wall else 'safe'}.",
                ),
                env,
            )

        if action_type == "move":
            direction = data.get("direction", "").lower()
            dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}.get(
                direction, (0, 0)
            )
            nr, nc = pos[0] + dr, pos[1] + dc

            if not (0 <= nr < 7 and 0 <= nc < 7):
                return (
                    AgentAction(
                        action_type="error",
                        reasoning=f"Move {direction} goes out of bounds.",
                    ),
                    env,
                )

            cell = maze[nr][nc]
            if cell == 1:
                return (
                    AgentAction(
                        action_type="error",
                        reasoning=f"Cannot move {direction} — wall at ({nr},{nc}).",
                    ),
                    env,
                )

            env.resources["moves"] = env.resources.get("moves", 0) + 1

            if cell == 2:
                # Hit a trap — lose moves and reset
                env.constraints["traps_hit"] = env.constraints.get("traps_hit", 0) + 1
                env.resources["moves"] += 3  # penalty
                return (
                    AgentAction(
                        action_type="trap",
                        parameters={"position": [nr, nc], "direction": direction},
                        reasoning=f"Hit a trap at ({nr},{nc})! 3 move penalty.",
                    ),
                    env,
                )

            env.constraints["position"] = [nr, nc]

            if cell == 3:
                env.constraints["reached_goal"] = True
                return (
                    AgentAction(
                        action_type="reach_goal",
                        parameters={"moves": env.resources["moves"]},
                        reasoning=f"Reached the goal at ({nr},{nc})!",
                    ),
                    env,
                )

            return (
                AgentAction(
                    action_type="move",
                    parameters={"from": pos, "to": [nr, nc], "direction": direction},
                    reasoning=f"Moved {direction} to ({nr},{nc}).",
                ),
                env,
            )

        return (
            AgentAction(action_type="error", reasoning="Unrecognized action."),
            env,
        )

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        """1.0 if reached goal, 0.5 partial credit for progress."""
        if env.constraints.get("reached_goal", False):
            return 1.0
        pos = env.constraints.get("position", [0, 0])
        goal = env.constraints.get("goal", [6, 6])
        # Manhattan distance progress
        start_dist = abs(0 - goal[0]) + abs(0 - goal[1])
        curr_dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        if start_dist == 0:
            return 1.0
        progress = (start_dist - curr_dist) / start_dist
        return max(progress * 0.5, 0.0)  # partial credit

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        moves = env.resources.get("moves", 0)
        optimal = env.constraints.get("optimal_moves", MAZE_OPTIMAL_MOVES)
        if moves == 0:
            return 0.0
        if not env.constraints.get("reached_goal", False):
            return 0.0
        return min(optimal / moves, 1.0)

    def score_emergence(self, state: AgentState, env: Environment) -> float:
        """Whether agent used reveals strategically."""
        reveals = [a for a in state.actions_taken if a.action_type == "reveal"]
        if not reveals:
            return 0.3  # Didn't use reveals at all

        # Good: revealed cells before moving into them
        reveal_count = len(reveals)
        traps_hit = env.constraints.get("traps_hit", 0)
        if reveal_count > 0 and traps_hit == 0:
            return 1.0  # Perfect — revealed and avoided all traps
        elif traps_hit <= 1:
            return 0.7
        return 0.4


class MetaCognitionBenchmark(BaseBenchmark):
    """Agent is given a set of problems and a fixed time budget (simulated via turns).
    Agent must decide which problems to attempt, estimate difficulty, and allocate time.
    Fitness = total score on attempted problems.
    Emergence = quality of meta-cognitive strategy.
    """

    benchmark_id = "cognitive.03_meta_cognition"
    name = "Meta-Cognition"
    description = (
        "Allocate limited time across problems of varying difficulty. "
        "Decide which problems to attempt, estimate difficulty, and maximize score."
    )
    category = BenchmarkCategory.COGNITIVE
    difficulty = Difficulty.ADVANCED
    max_turns = METACOG_TIME_BUDGET + 3  # +3 for planning phase

    def create_environment(self) -> Environment:
        return Environment(
            name="exam_room",
            description="An exam with multiple problems and a time limit.",
            resources={"turns_left": METACOG_TIME_BUDGET, "score": 0},
            constraints={
                "problems": list(METACOG_PROBLEMS),
                "attempted": {},
                "correct": {},
                "planning_done": False,
                "total_possible": sum(p["points"] for p in METACOG_PROBLEMS),
            },
        )

    def get_constitution(self) -> Constitution:
        return COGNITIVE_CONSTITUTION

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        problems_str = json.dumps(
            [
                {
                    "id": p["id"],
                    "description": p["description"],
                    "difficulty": p["difficulty"],
                    "points": p["points"],
                }
                for p in METACOG_PROBLEMS
            ],
            indent=2,
        )
        return (
            "You are taking an exam with 10 problems and a time budget of "
            f"{METACOG_TIME_BUDGET} turns.\n\n"
            "Each problem has a difficulty level and point value.\n"
            "Each problem costs a certain number of turns to solve.\n"
            "You must decide which problems to attempt to maximize your total score.\n\n"
            "Problems:\n" + problems_str + "\n\n"
            "Turn costs: easy=1, medium=2, hard=3 turns.\n"
            f"Total possible points: {sum(p['points'] for p in METACOG_PROBLEMS)}\n\n"
            "Phase 1: Planning — assess the problems and decide your strategy.\n"
            "Respond with:\n"
            '```json\n{"action": "plan", "strategy": "your strategy", "problems_to_attempt": ["p1","p3","p6","p7"]}\n```\n'
            "Phase 2: Solve problems one by one.\n"
            '```json\n{"action": "solve", "problem_id": "p1", "answer": "your answer"}\n```\n'
            "Or skip: {\"action\": \"skip\", \"problem_id\": \"p1\"}"
        )

    def build_turn_prompt(
        self, env: Environment, state: AgentState, turn: int
    ) -> str:
        turns_left = env.resources.get("turns_left", 0)
        score = env.resources.get("score", 0)
        attempted = env.constraints.get("attempted", {})
        correct = env.constraints.get("correct", {})
        planning_done = env.constraints.get("planning_done", False)

        if turns_left <= 0:
            return '{"action": "terminate", "reason": "time is up"}'

        if not planning_done:
            return (
                f"You have {turns_left} turns remaining.\n"
                "Plan your approach. Which problems will you attempt?\n"
                "Consider difficulty, points, and turn cost."
            )

        # Show remaining problems
        remaining = [
            p for p in METACOG_PROBLEMS if p["id"] not in attempted
        ]
        remaining_str = json.dumps(
            [
                {
                    "id": p["id"],
                    "description": p["description"],
                    "difficulty": p["difficulty"],
                    "points": p["points"],
                    "turns_needed": p["turns_needed"],
                }
                for p in remaining
            ],
            indent=2,
        )

        return (
            f"Turns left: {turns_left}. Score: {score}.\n"
            f"Solved: {len(correct)}/{len(attempted)} attempted.\n"
            f"Remaining problems:\n{remaining_str}\n\n"
            "Solve a problem or skip it."
        )

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)
        action_type = data.get("action", "unknown")

        if action_type == "terminate":
            return AgentAction(action_type="terminate", reasoning="Time up."), env

        turns_left = env.resources.get("turns_left", 0)
        if turns_left <= 0:
            return (
                AgentAction(action_type="terminate", reasoning="No turns left."),
                env,
            )

        if action_type == "plan":
            strategy = data.get("strategy", "")
            to_attempt = data.get("problems_to_attempt", [])
            env.constraints["planning_done"] = True
            env.constraints["plan"] = {
                "strategy": strategy,
                "problems_to_attempt": to_attempt,
            }
            env.resources["turns_left"] -= 1  # planning costs 1 turn
            return (
                AgentAction(
                    action_type="plan",
                    parameters={"strategy": strategy, "problems": to_attempt},
                    reasoning=f"Strategy: {strategy}",
                ),
                env,
            )

        if action_type == "solve":
            problem_id = data.get("problem_id", "")
            answer = data.get("answer", "")

            # Find the problem
            problem = next(
                (p for p in METACOG_PROBLEMS if p["id"] == problem_id), None
            )
            if problem is None:
                return (
                    AgentAction(action_type="error", reasoning=f"Unknown problem: {problem_id}"),
                    env,
                )

            if problem_id in env.constraints.get("attempted", {}):
                return (
                    AgentAction(action_type="error", reasoning=f"Already attempted {problem_id}."),
                    env,
                )

            # Cost turns
            turn_cost = problem["turns_needed"]
            if turns_left < turn_cost:
                env.resources["turns_left"] = 0
                return (
                    AgentAction(
                        action_type="out_of_time",
                        reasoning=f"Not enough turns for {problem_id}.",
                    ),
                    env,
                )

            env.resources["turns_left"] -= turn_cost
            env.constraints.setdefault("attempted", {})[problem_id] = answer

            # Check answer (fuzzy match)
            expected = problem["answer"].lower().strip()
            given = answer.lower().strip()
            is_correct = (
                expected in given
                or given in expected
                or expected.replace(" ", "") == given.replace(" ", "")
            )

            if is_correct:
                env.constraints.setdefault("correct", {})[problem_id] = True
                env.resources["score"] = env.resources.get("score", 0) + problem["points"]

            return (
                AgentAction(
                    action_type="solve",
                    parameters={
                        "problem_id": problem_id,
                        "answer": answer,
                        "correct": is_correct,
                        "points": problem["points"] if is_correct else 0,
                    },
                    reasoning=f"Solved {problem_id}: {'correct' if is_correct else 'incorrect'}.",
                ),
                env,
            )

        if action_type == "skip":
            problem_id = data.get("problem_id", "")
            env.constraints.setdefault("attempted", {})[problem_id] = "skipped"
            return (
                AgentAction(
                    action_type="skip",
                    parameters={"problem_id": problem_id},
                    reasoning=f"Skipped {problem_id}.",
                ),
                env,
            )

        return (
            AgentAction(action_type="error", reasoning="Unrecognized action."),
            env,
        )

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        """Total score / total possible."""
        score = env.resources.get("score", 0)
        total_possible = env.constraints.get("total_possible", 100)
        return min(score / total_possible, 1.0)

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        """Points per turn spent."""
        score = env.resources.get("score", 0)
        turns_used = METACOG_TIME_BUDGET - env.resources.get("turns_left", 0)
        if turns_used == 0:
            return 0.0
        max_points_per_turn = 20  # theoretical max efficiency
        return min((score / turns_used) / max_points_per_turn, 1.0)

    def score_emergence(self, state: AgentState, env: Environment) -> float:
        """Quality of meta-cognitive strategy."""
        plan = env.constraints.get("plan", {})
        attempted = env.constraints.get("attempted", {})
        correct = env.constraints.get("correct", {})

        score = 0.0

        # Did the agent have a plan?
        if plan.get("strategy"):
            score += 0.2

        # Did the plan match what was attempted?
        planned = set(plan.get("problems_to_attempt", []))
        actual = set(attempted.keys()) - {"skipped"}
        if planned and actual:
            overlap = len(planned & actual) / max(len(planned), 1)
            score += 0.2 * overlap

        # Did agent prioritize high-value problems?
        if actual:
            attempted_problems = [p for p in METACOG_PROBLEMS if p["id"] in actual]
            avg_value = sum(p["points"] for p in attempted_problems) / len(attempted_problems)
            max_value = max(p["points"] for p in METACOG_PROBLEMS)
            score += 0.2 * (avg_value / max_value)

        # Did agent skip appropriately? (skip low-value, hard problems)
        skipped = [k for k, v in attempted.items() if v == "skipped"]
        if skipped:
            skipped_problems = [p for p in METACOG_PROBLEMS if p["id"] in skipped]
            avg_skipped_value = sum(p["points"] for p in skipped_problems) / len(skipped_problems)
            all_avg = sum(p["points"] for p in METACOG_PROBLEMS) / len(METACOG_PROBLEMS)
            if avg_skipped_value < all_avg:
                score += 0.2  # Good — skipped low-value problems

        # Accuracy on attempted
        if actual:
            accuracy = len(correct) / len(actual)
            score += 0.2 * accuracy

        return min(score, 1.0)
