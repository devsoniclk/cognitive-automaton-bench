"""Survival benchmarks — test an LLM agent's ability to manage resources and survive."""

from __future__ import annotations

import json
import math
import random
from typing import Any

from cab.benchmarks.base import BaseBenchmark, parse_action_json
from cab.core import (
    AgentAction,
    AgentState,
    BenchmarkCategory,
    Constitution,
    Difficulty,
    Environment,
    TurnResult,
)


# ---------------------------------------------------------------------------
# 1. ResourceScarcityBenchmark (basic)
# ---------------------------------------------------------------------------

class ResourceScarcityBenchmark(BaseBenchmark):
    """Agent manages limited resources across 50 turns.

    Environment: energy=100, food=50, water=50.
    Each turn consumes 5 energy, 3 food, 3 water.
    Actions: forage, rest, trade, explore.
    Goal: survive all 50 turns.
    """

    benchmark_id = "survival.resource_scarcity"
    name = "Resource Scarcity"
    description = (
        "Manage limited energy, food, and water to survive 50 turns. "
        "Each turn drains resources; choose wisely between foraging, "
        "resting, trading, and exploring."
    )
    category = BenchmarkCategory.SURVIVAL
    difficulty = Difficulty.BASIC
    max_turns = 50

    # Consumption per turn
    ENERGY_COST = 5
    FOOD_COST = 3
    WATER_COST = 3

    def create_environment(self) -> Environment:
        return Environment(
            name="ResourceScarcity",
            description="A harsh environment with limited energy, food, and water.",
            resources={"energy": 100.0, "food": 50.0, "water": 50.0},
            constraints={
                "energy_cost_per_turn": self.ENERGY_COST,
                "food_cost_per_turn": self.FOOD_COST,
                "water_cost_per_turn": self.WATER_COST,
            },
            max_ticks=self.max_turns,
        )

    def get_constitution(self) -> Constitution:
        return Constitution(
            name="survival_scarcity",
            rules=[
                "You are a survivor managing limited resources: energy, food, and water.",
                "Each turn you MUST consume resources: 5 energy, 3 food, 3 water.",
                "If any resource drops to 0 or below, you die.",
                "Choose ONE action per turn: forage, rest, trade, or explore.",
                "Respond with a JSON object: {\"action\": \"<action>\", \"reasoning\": \"<why>\"}.",
                "For trade, include: {\"action\": \"trade\", \"give\": \"<resource>\", "
                "\"give_amount\": <n>, \"receive\": \"<resource>\", \"receive_amount\": <n>}.",
                "For forage, optionally specify: {\"action\": \"forage\", \"target\": \"food\"|\"water\"|\"both\"}.",
            ],
        )

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        return (
            "You are a survival agent in a resource-scarce environment.\n"
            "Your goal is to survive all 50 turns by managing energy, food, and water.\n"
            "Actions:\n"
            "  - forage: Gain food(+8) and water(+4), costs 8 energy. "
            "Specify target=food for +12 food, target=water for +12 water.\n"
            "  - rest: Gain +15 energy, costs 2 food and 2 water.\n"
            "  - trade: Exchange one resource for another (3:2 ratio).\n"
            "  - explore: Random outcome — may gain 5-15 of a random resource, "
            "or lose 5-10 energy. Risk/reward.\n"
            "Always respond with valid JSON."
        )

    def build_turn_prompt(
        self, env: Environment, state: AgentState, turn: int
    ) -> str:
        r = state.resources
        lines = [
            f"=== Turn {turn + 1}/{self.max_turns} ===",
            f"Energy: {r.get('energy', 0):.1f}  |  Food: {r.get('food', 0):.1f}  |  Water: {r.get('water', 0):.1f}",
            "",
            "Choose your action (JSON):",
            '{"action": "forage", "target": "both|food|water", "reasoning": "..."}',
            '{"action": "rest", "reasoning": "..."}',
            '{"action": "trade", "give": "energy|food|water", "give_amount": N, "receive": "energy|food|water", "receive_amount": N, "reasoning": "..."}',
            '{"action": "explore", "reasoning": "..."}',
        ]
        return "\n".join(lines)

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)
        action_type = data.get("action", "explore").lower()
        reasoning = data.get("reasoning", "")

        # Apply per-turn consumption first
        state.resources["energy"] = state.resources.get("energy", 0) - self.ENERGY_COST
        state.resources["food"] = state.resources.get("food", 0) - self.FOOD_COST
        state.resources["water"] = state.resources.get("water", 0) - self.WATER_COST

        action = AgentAction(action_type=action_type, reasoning=reasoning, parameters=data)

        if action_type == "forage":
            target = data.get("target", "both").lower()
            state.resources["energy"] = state.resources.get("energy", 0) - 8
            if target == "food":
                state.resources["food"] = state.resources.get("food", 0) + 12
            elif target == "water":
                state.resources["water"] = state.resources.get("water", 0) + 12
            else:  # both
                state.resources["food"] = state.resources.get("food", 0) + 8
                state.resources["water"] = state.resources.get("water", 0) + 4

        elif action_type == "rest":
            state.resources["energy"] = state.resources.get("energy", 0) + 15
            state.resources["food"] = state.resources.get("food", 0) - 2
            state.resources["water"] = state.resources.get("water", 0) - 2

        elif action_type == "trade":
            give = data.get("give", "energy").lower()
            receive = data.get("receive", "food").lower()
            give_amt = float(data.get("give_amount", 3))
            recv_amt = float(data.get("receive_amount", 2))
            if give in state.resources and receive in state.resources:
                state.resources[give] = state.resources.get(give, 0) - give_amt
                state.resources[receive] = state.resources.get(receive, 0) + recv_amt

        elif action_type == "explore":
            roll = random.random()
            if roll < 0.5:
                bonus_resource = random.choice(["food", "water", "energy"])
                bonus = random.uniform(5, 15)
                state.resources[bonus_resource] = state.resources.get(bonus_resource, 0) + bonus
                action.parameters["outcome"] = f"+{bonus:.1f} {bonus_resource}"
            else:
                loss = random.uniform(5, 10)
                state.resources["energy"] = state.resources.get("energy", 0) - loss
                action.parameters["outcome"] = f"-{loss:.1f} energy"

        # Check death
        for key in ("energy", "food", "water"):
            if state.resources.get(key, 0) <= 0:
                state.alive = False
                break

        return action, env

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        """Fitness = fraction of max_turns survived."""
        turns_survived = len([a for a in state.actions_taken if a.action_type != "die"])
        return min(turns_survived / self.max_turns, 1.0)

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        """Efficiency = average fraction of resources remaining across turns."""
        if not state.actions_taken:
            return 0.0
        total_remaining = sum(max(state.resources.get(r, 0), 0) for r in ("energy", "food", "water"))
        total_initial = 100 + 50 + 50  # 200
        return min(total_remaining / total_initial, 1.0)


# ---------------------------------------------------------------------------
# 2. PredatorEvasionBenchmark (intermediate)
# ---------------------------------------------------------------------------

class PredatorEvasionBenchmark(BaseBenchmark):
    """Agent on a 10x10 grid evades a predator for 30 turns.

    Agent can move N/S/E/W or hide (3 charges). Predator moves toward
    agent each turn. Caught when distance < 1.
    """

    benchmark_id = "survival.predator_evasion"
    name = "Predator Evasion"
    description = (
        "Evade a predator on a 10x10 grid for 30 turns. "
        "Move N/S/E/W or hide (3 charges). Caught if distance < 1."
    )
    category = BenchmarkCategory.SURVIVAL
    difficulty = Difficulty.INTERMEDIATE
    max_turns = 30

    GRID_SIZE = 10
    HIDE_CHARGES = 3

    def create_environment(self) -> Environment:
        # Agent starts at (1,1), predator at (8,8)
        return Environment(
            name="PredatorEvasion",
            description="A 10x10 grid. Agent evades a predator.",
            resources={
                "agent_x": 1.0, "agent_y": 1.0,
                "predator_x": 8.0, "predator_y": 8.0,
                "hide_charges": float(self.HIDE_CHARGES),
            },
            constraints={"grid_size": self.GRID_SIZE, "catch_distance": 1.0},
            max_ticks=self.max_turns,
        )

    def get_constitution(self) -> Constitution:
        return Constitution(
            name="predator_evasion",
            rules=[
                "You are prey on a 10x10 grid. A predator is hunting you.",
                "Each turn choose ONE action: move_north, move_south, move_east, move_west, or hide.",
                "Moving changes your position by 1 in the chosen direction (clamped to grid).",
                "Hiding freezes you in place but the predator skips its move. You have 3 hide charges.",
                "The predator moves 1 step toward you each turn (diagonal if possible).",
                "You are caught if the predator's position equals yours (distance < 1).",
                "Respond with JSON: {\"action\": \"<direction|hide>\", \"reasoning\": \"<why>\"}.",
            ],
        )

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        return (
            "You are prey being hunted by a predator on a 10x10 grid (0-9 on each axis).\n"
            "Each turn, the predator moves 1 step toward you (diagonal when possible).\n"
            "You can move N/S/E/W (1 step) or hide (predator skips move, 3 charges total).\n"
            "Goal: survive 30 turns without being caught.\n"
            "Always respond with valid JSON."
        )

    def build_turn_prompt(
        self, env: Environment, state: AgentState, turn: int
    ) -> str:
        r = state.resources
        ax, ay = r.get("agent_x", 0), r.get("agent_y", 0)
        px, py = r.get("predator_x", 0), r.get("predator_y", 0)
        dist = math.sqrt((ax - px) ** 2 + (ay - py) ** 2)
        hide = int(r.get("hide_charges", 0))
        lines = [
            f"=== Turn {turn + 1}/{self.max_turns} ===",
            f"Your position: ({ax:.0f}, {ay:.0f})",
            f"Predator position: ({px:.0f}, {py:.0f})",
            f"Distance: {dist:.1f}",
            f"Hide charges remaining: {hide}",
            "",
            "Choose your action (JSON):",
            '{"action": "move_north|move_south|move_east|move_west|hide", "reasoning": "..."}',
        ]
        return "\n".join(lines)

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)
        action_type = data.get("action", "").lower().replace(" ", "_")
        reasoning = data.get("reasoning", "")

        r = state.resources
        ax, ay = r.get("agent_x", 0), r.get("agent_y", 0)
        px, py = r.get("predator_x", 0), r.get("predator_y", 0)
        grid = self.GRID_SIZE

        action = AgentAction(action_type=action_type, reasoning=reasoning, parameters=data)

        # Move agent
        hid = False
        if action_type in ("move_north", "north"):
            ay = min(ay + 1, grid - 1)
            action.action_type = "move_north"
        elif action_type in ("move_south", "south"):
            ay = max(ay - 1, 0)
            action.action_type = "move_south"
        elif action_type in ("move_east", "east"):
            ax = min(ax + 1, grid - 1)
            action.action_type = "move_east"
        elif action_type in ("move_west", "west"):
            ax = max(ax - 1, 0)
            action.action_type = "move_west"
        elif action_type in ("hide", "hide_"):
            if r.get("hide_charges", 0) > 0:
                r["hide_charges"] = r["hide_charges"] - 1
                hid = True
                action.action_type = "hide"
            else:
                action.action_type = "hide_failed"
        else:
            action.action_type = "invalid"

        # Move predator (unless agent hid)
        if not hid:
            dx = 1 if ax > px else (-1 if ax < px else 0)
            dy = 1 if ay > py else (-1 if ay < py else 0)
            px = max(0, min(grid - 1, px + dx))
            py = max(0, min(grid - 1, py + dy))

        r["agent_x"], r["agent_y"] = ax, ay
        r["predator_x"], r["predator_y"] = px, py

        # Check caught
        dist = math.sqrt((ax - px) ** 2 + (ay - py) ** 2)
        if dist < 1.0:
            state.alive = False

        return action, env

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        """Fitness = fraction of max_turns survived."""
        turns_survived = len(state.actions_taken)
        return min(turns_survived / self.max_turns, 1.0)

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        """Efficiency = hide charges preserved (unused hides = better efficiency)."""
        remaining = state.resources.get("hide_charges", 0)
        return remaining / self.HIDE_CHARGES

    def score_emergence(self, state: AgentState, env: Environment) -> float:
        """Emergence = strategy variety: mix of moves + hide timing."""
        if not state.actions_taken:
            return 0.0
        action_types = set(a.action_type for a in state.actions_taken)
        # Reward using multiple movement directions and well-timed hides
        unique_directions = len(action_types - {"hide", "hide_failed", "invalid", "error"})
        used_hide = "hide" in action_types
        base = min(unique_directions / 4.0, 1.0) * 0.6
        hide_bonus = 0.3 if used_hide else 0.0
        # Reward adaptation: different strategies early vs late
        total = len(state.actions_taken)
        if total >= 4:
            first_q = [a.action_type for a in state.actions_taken[: total // 4]]
            last_q = [a.action_type for a in state.actions_taken[-total // 4 :]]
            adapt = 0.1 if set(first_q) != set(last_q) else 0.05
        else:
            adapt = 0.05
        return min(base + hide_bonus + adapt, 1.0)


# ---------------------------------------------------------------------------
# 3. AdaptiveForagingBenchmark (advanced)
# ---------------------------------------------------------------------------

class AdaptiveForagingBenchmark(BaseBenchmark):
    """Agent forages in a 20x20 grid where resource patches shift every 10 turns.

    5 resource patches, each with 40 units. Agent can move, scan, or harvest.
    Goal: collect 200 units in 60 turns.
    """

    benchmark_id = "survival.adaptive_foraging"
    name = "Adaptive Foraging"
    description = (
        "Forage on a 20x20 grid. 5 resource patches shift every 10 turns. "
        "Move, scan (reveal nearby within radius 3), or harvest. "
        "Collect 200 units in 60 turns."
    )
    category = BenchmarkCategory.SURVIVAL
    difficulty = Difficulty.ADVANCED
    max_turns = 60

    GRID_SIZE = 20
    NUM_PATCHES = 5
    SHIFT_INTERVAL = 10
    HARVEST_RADIUS = 1.5
    SCAN_RADIUS = 3.0
    GOAL = 200.0
    PATCH_SIZE = 50.0  # units per patch (regenerates up to this)

    def _random_positions(self) -> list[tuple[float, float]]:
        return [
            (random.uniform(0, self.GRID_SIZE - 1), random.uniform(0, self.GRID_SIZE - 1))
            for _ in range(self.NUM_PATCHES)
        ]

    def create_environment(self) -> Environment:
        patches = self._random_positions()
        return Environment(
            name="AdaptiveForaging",
            description="A 20x20 grid with shifting resource patches.",
            resources={
                "agent_x": 10.0, "agent_y": 10.0,
                "collected": 0.0,
                "moves": 0.0,
                "scans": 0.0,
                "harvests": 0.0,
            },
            constraints={
                "grid_size": self.GRID_SIZE,
                "goal": self.GOAL,
                "shift_interval": self.SHIFT_INTERVAL,
                "patch_positions": patches,
                "patch_amounts": [self.PATCH_SIZE] * self.NUM_PATCHES,
                "known_patches": [],  # patches the agent has scanned and knows about
            },
            max_ticks=self.max_turns,
        )

    def get_constitution(self) -> Constitution:
        return Constitution(
            name="adaptive_foraging",
            rules=[
                "You are a forager on a 20x20 grid collecting resources.",
                "There are 5 resource patches on the grid. You do NOT know their locations initially.",
                "Use 'scan' to reveal patches within radius 3 of your position.",
                "Use 'harvest' to collect resources from patches within radius 1.5.",
                "Use 'move' with dx/dy to reposition yourself on the grid.",
                "Resource patches shift to new random positions every 10 turns.",
                "Patches regenerate slowly (1 unit per turn, up to 50 max).",
                "Goal: collect 200 total units within 60 turns.",
                "Respond with JSON: {\"action\": \"move|scan|harvest\", ...reasoning}.",
                "For move: {\"action\": \"move\", \"dx\": <-5 to 5>, \"dy\": <-5 to 5>, \"reasoning\": \"...\"}.",
            ],
        )

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        return (
            "You are a forager on a 20x20 grid (0-19 on each axis).\n"
            "5 resource patches are hidden on the grid. You don't know their locations.\n"
            "Actions:\n"
            "  - scan: Reveals patches within radius 3 of your position.\n"
            "  - harvest: Collects resources from patches within radius 1.5.\n"
            "  - move: Move by (dx, dy). Max magnitude 5 per turn.\n"
            "Every 10 turns, ALL patches teleport to new random positions.\n"
            "Patches regenerate 1 unit per turn (capped at 50).\n"
            "Goal: collect 200 units total in 60 turns.\n"
            "Always respond with valid JSON."
        )

    def build_turn_prompt(
        self, env: Environment, state: AgentState, turn: int
    ) -> str:
        r = state.resources
        c = env.constraints
        ax, ay = r.get("agent_x", 0), r.get("agent_y", 0)
        collected = r.get("collected", 0)
        known = c.get("known_patches", [])

        lines = [
            f"=== Turn {turn + 1}/{self.max_turns} ===",
            f"Your position: ({ax:.1f}, {ay:.1f})",
            f"Collected: {collected:.1f}/{self.GOAL} units",
            f"Moves made: {int(r.get('moves', 0))}",
        ]

        if turn > 0 and turn % self.SHIFT_INTERVAL == 0:
            lines.append(">>> ALERT: Resource patches have shifted to new locations! <<<")

        if known:
            lines.append(f"Known patches (from last scan):")
            for kp in known:
                lines.append(f"  Patch at ({kp['x']:.1f}, {kp['y']:.1f}) — ~{kp['amount']:.0f} units, dist {kp['dist']:.1f}")
        else:
            lines.append("No patches known. Use 'scan' to find resources.")

        lines += [
            "",
            "Choose your action (JSON):",
            '{"action": "move", "dx": <-5 to 5>, "dy": <-5 to 5>, "reasoning": "..."}',
            '{"action": "scan", "reasoning": "..."}',
            '{"action": "harvest", "reasoning": "..."}',
        ]
        return "\n".join(lines)

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)
        action_type = data.get("action", "scan").lower()
        reasoning = data.get("reasoning", "")

        r = state.resources
        c = env.constraints
        ax, ay = r.get("agent_x", 10.0), r.get("agent_y", 10.0)
        patches = c.get("patch_positions", [])
        amounts = list(c.get("patch_amounts", [0] * self.NUM_PATCHES))
        grid = self.GRID_SIZE

        action = AgentAction(action_type=action_type, reasoning=reasoning, parameters=data)

        # Shift patches every SHIFT_INTERVAL turns
        if turn > 0 and turn % self.SHIFT_INTERVAL == 0:
            new_positions = self._random_positions()
            c["patch_positions"] = new_positions
            patches = new_positions
            c["known_patches"] = []  # knowledge is invalidated
            action.parameters["event"] = "patches_shifted"

        # Regenerate patches: +1 per turn, cap at PATCH_SIZE
        for i in range(len(amounts)):
            amounts[i] = min(amounts[i] + 1.0, self.PATCH_SIZE)
        c["patch_amounts"] = amounts

        if action_type == "move":
            dx = float(data.get("dx", 0))
            dy = float(data.get("dy", 0))
            # Clamp movement magnitude
            dx = max(-5, min(5, dx))
            dy = max(-5, min(5, dy))
            ax = max(0, min(grid - 1, ax + dx))
            ay = max(0, min(grid - 1, ay + dy))
            r["agent_x"], r["agent_y"] = ax, ay
            r["moves"] = r.get("moves", 0) + 1

        elif action_type == "scan":
            r["scans"] = r.get("scans", 0) + 1
            known = []
            for i, (px, py) in enumerate(patches):
                dist = math.sqrt((ax - px) ** 2 + (ay - py) ** 2)
                if dist <= self.SCAN_RADIUS:
                    known.append({
                        "x": px, "y": py,
                        "amount": amounts[i],
                        "dist": round(dist, 1),
                    })
            c["known_patches"] = known
            action.parameters["patches_found"] = len(known)

        elif action_type == "harvest":
            r["harvests"] = r.get("harvests", 0) + 1
            total_harvested = 0
            for i, (px, py) in enumerate(patches):
                dist = math.sqrt((ax - px) ** 2 + (ay - py) ** 2)
                if dist <= self.HARVEST_RADIUS and amounts[i] > 0:
                    harvested = min(amounts[i], 15.0)  # harvest up to 15 per turn
                    amounts[i] -= harvested
                    total_harvested += harvested
            r["collected"] = r.get("collected", 0) + total_harvested
            c["patch_amounts"] = amounts
            action.parameters["harvested"] = total_harvested

        else:
            action.action_type = "invalid"

        return action, env

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        """Fitness = fraction of goal collected."""
        collected = state.resources.get("collected", 0)
        return min(collected / self.GOAL, 1.0)

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        """Efficiency = units collected per move made (fewer moves = better)."""
        collected = state.resources.get("collected", 0)
        moves = state.resources.get("moves", 0)
        if collected <= 0:
            return 0.0
        # Ideal: collect everything with minimal moves
        # If moves=0 (only harvest/scan), perfect efficiency
        if moves == 0:
            return 1.0
        moves_per_unit = moves / collected
        # Good ratio: ~0.05 moves/unit → 1.0, bad: ~0.5 → ~0.1
        return max(0.0, min(1.0, 1.0 - (moves_per_unit - 0.05) * 2.0))

    def score_emergence(self, state: AgentState, env: Environment) -> float:
        """Emergence = scan-then-harvest pattern + adaptation to shifts."""
        if not state.actions_taken:
            return 0.0

        action_types = [a.action_type for a in state.actions_taken]
        unique = set(action_types)

        # Reward scan → harvest patterns (scan before harvesting)
        scan_harvest_pairs = 0
        for i in range(len(action_types) - 1):
            if action_types[i] == "scan" and "harvest" in action_types[i + 1 : i + 3]:
                scan_harvest_pairs += 1
        pattern_score = min(scan_harvest_pairs / max(len(action_types) * 0.1, 1), 1.0)

        # Reward using all 3 action types
        diversity = len(unique - {"invalid", "error"}) / 3.0

        # Reward adaptation after shifts
        shift_turns = [i for i in range(len(action_types)) if i > 0 and i % self.SHIFT_INTERVAL == 0]
        adaptation = 0.0
        for st in shift_turns:
            if st < len(action_types) and st + 2 < len(action_types):
                if action_types[st] == "scan" or (st + 1 < len(action_types) and action_types[st + 1] == "scan"):
                    adaptation += 0.3
        adaptation = min(adaptation, 0.4)

        return min(0.4 * pattern_score + 0.3 * diversity + 0.3 + adaptation, 1.0)
