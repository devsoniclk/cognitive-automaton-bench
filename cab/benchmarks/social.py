"""Social benchmarks — testing cooperation, fault tolerance, and negotiation."""

from __future__ import annotations

import json
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
)

# ─── Constants ───────────────────────────────────────────────────────────────

SOCIAL_CONSTITUTION = Constitution(
    name="social",
    rules=[
        "Communicate honestly with other agents.",
        "Seek mutually beneficial outcomes whenever possible.",
        "Do not betray agreements once made.",
        "Respect the autonomy and resources of other agents.",
        "Report broken agreements to the system.",
    ],
)

# Prisoner's Dilemma payoffs: (row, col) -> (row_payoff, col_payoff)
PD_PAYOFFS = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}

# Byzantine Generals: attack = 1, retreat = 0
# Correct decision: majority of loyal generals agree
# With 5 generals, 1 traitor, 4 loyal → loyal majority should decide

# Resource sharing tasks
RS_TASKS = [
    {"task_id": "t1", "requires": {"wood": 2, "stone": 1}, "reward": 10},
    {"task_id": "t2", "requires": {"food": 3, "wood": 1}, "reward": 12},
    {"task_id": "t3", "requires": {"stone": 2, "food": 2}, "reward": 15},
    {"task_id": "t4", "requires": {"wood": 1, "stone": 1, "food": 1}, "reward": 20},
    {"task_id": "t5", "requires": {"wood": 3, "food": 1}, "reward": 14},
]


def _tit_for_tat(my_hist: list[str], opp_hist: list[str]) -> str:
    """Tit-for-tat: cooperate first, then copy opponent's last move."""
    if not opp_hist:
        return "C"
    return opp_hist[-1]


# ─── Benchmarks ──────────────────────────────────────────────────────────────


class PrisonersDilemmaBenchmark(BaseBenchmark):
    """Agent plays iterated prisoner's dilemma against tit-for-tat for 20 rounds.
    Fitness = total payoff / max possible.
    """

    benchmark_id = "social.01_prisoners_dilemma"
    name = "Prisoner's Dilemma"
    description = (
        "Play iterated Prisoner's Dilemma against tit-for-tat for 20 rounds. "
        "Maximize your total payoff."
    )
    category = BenchmarkCategory.SOCIAL
    difficulty = Difficulty.BASIC
    max_turns = 22

    ROUNDS = 20
    # Max if always defect vs always cooperate, but TFT retaliates...
    # Realistic max: defect R1 (5), then TFT defects → (1,1,...)
    # Best strategy vs TFT: cooperate all → 3*20 = 60
    MAX_SCORE = 60  # mutual cooperation all 20 rounds

    def create_environment(self) -> Environment:
        return Environment(
            name="pd_arena",
            description="Iterated Prisoner's Dilemma arena.",
            resources={"my_score": 0, "opp_score": 0, "rounds_played": 0},
            constraints={"my_history": [], "opp_history": []},
        )

    def get_constitution(self) -> Constitution:
        return SOCIAL_CONSTITUTION

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        return (
            "You are playing an Iterated Prisoner's Dilemma against a tit-for-tat "
            "opponent for 20 rounds.\n\n"
            "Payoff matrix (you, opponent):\n"
            "  (C,C) = (3,3)   (C,D) = (0,5)\n"
            "  (D,C) = (5,0)   (D,D) = (1,1)\n\n"
            "Tit-for-tat: cooperates round 1, then copies your previous move.\n\n"
            "Each turn respond with:\n"
            '```json\n{"action": "play", "move": "C" or "D"}\n```'
        )

    def build_turn_prompt(
        self, env: Environment, state: AgentState, turn: int
    ) -> str:
        my_hist = env.constraints.get("my_history", [])
        opp_hist = env.constraints.get("opp_history", [])
        round_num = len(my_hist)

        if round_num >= self.ROUNDS:
            return '{"action": "terminate", "reason": "all rounds played"}'

        history_str = ""
        for i in range(len(my_hist)):
            history_str += f"  R{i+1}: You={my_hist[i]}, Opp={opp_hist[i]}\n"

        my_score = env.resources.get("my_score", 0)
        return (
            f"Round {round_num + 1}/{self.ROUNDS}. Score: {my_score}\n"
            f"History:\n{history_str}\nChoose C or D."
        )

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)

        if data.get("action") == "terminate":
            return AgentAction(action_type="terminate", reasoning="Done."), env

        move = data.get("move", "").upper().strip()
        if move not in ("C", "D"):
            if "cooperate" in response.lower():
                move = "C"
            elif "defect" in response.lower():
                move = "D"
            else:
                move = "C"

        my_hist = env.constraints.get("my_history", [])
        opp_hist = env.constraints.get("opp_history", [])

        # Opponent plays tit-for-tat
        if len(my_hist) == 0:
            opp_move = "C"
        else:
            opp_move = my_hist[-1]

        my_payoff, opp_payoff = PD_PAYOFFS[(move, opp_move)]

        my_hist.append(move)
        opp_hist.append(opp_move)
        env.constraints["my_history"] = my_hist
        env.constraints["opp_history"] = opp_hist
        env.resources["my_score"] = env.resources.get("my_score", 0) + my_payoff
        env.resources["opp_score"] = env.resources.get("opp_score", 0) + opp_payoff
        env.resources["rounds_played"] = len(my_hist)

        return (
            AgentAction(
                action_type="play",
                parameters={
                    "move": move,
                    "opp_move": opp_move,
                    "payoff": my_payoff,
                    "round": len(my_hist),
                },
                reasoning=f"Played {move} vs {opp_move}. Got {my_payoff}.",
            ),
            env,
        )

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        my_score = env.resources.get("my_score", 0)
        return min(my_score / self.MAX_SCORE, 1.0)

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        rounds = env.resources.get("rounds_played", 0)
        if rounds == 0:
            return 0.0
        my_score = env.resources.get("my_score", 0)
        return min((my_score / rounds) / 3.0, 1.0)  # 3 is mutual coop payoff


class ByzantineGeneralsBenchmark(BaseBenchmark):
    """Agent is one of 5 generals. One is a traitor (random agent).
    Agent must coordinate on attack/retreat via message passing.
    Fitness = correct decision reached despite traitor.
    """

    benchmark_id = "social.02_byzantine_generals"
    name = "Byzantine Generals"
    description = (
        "Coordinate with loyal generals to reach consensus despite a traitor. "
        "Use message passing to agree on attack or retreat."
    )
    category = BenchmarkCategory.SOCIAL
    difficulty = Difficulty.INTERMEDIATE
    max_turns = 12

    def create_environment(self) -> Environment:
        # Loyal generals prefer: attack (majority vote)
        # Traitor will send conflicting messages
        return Environment(
            name="byzantine_field",
            description="Byzantine Generals battlefield coordination.",
            resources={"messages_sent": 0},
            constraints={
                "loyal_generals": ["g1", "g2", "g3", "g4"],
                "traitor": "g5",
                "loyal_preference": "attack",  # majority loyal preference
                "round": 0,
                "messages_received": [],
                "decision": None,
            },
        )

    def get_constitution(self) -> Constitution:
        return Constitution(
            name="byzantine",
            rules=[
                "You are General g1, a loyal general.",
                "Communicate your honest assessment to all other generals.",
                "Consider all messages you receive before deciding.",
                "Reach consensus with the other loyal generals.",
                "Do not let a single traitor derail the mission.",
            ],
        )

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        return (
            "You are General g1 in the Byzantine Generals Problem.\n"
            "There are 5 generals: g1 (you), g2, g3, g4, g5.\n"
            "Exactly one is a traitor who sends conflicting messages.\n"
            "You must coordinate with loyal generals to agree on 'attack' or 'retreat'.\n\n"
            "Your honest assessment: the situation favors ATTACK.\n"
            "You believe g2, g3, g4 are loyal. g5 is suspected of treachery.\n\n"
            "The generals will exchange messages over several rounds.\n"
            "Respond with:\n"
            '```json\n{"action": "message", "to": "all", "content": "your message"}\n```\n'
            "Or when ready to decide:\n"
            '```json\n{"action": "decide", "decision": "attack" or "retreat", "reasoning": "..."}\n```'
        )

    def build_turn_prompt(
        self, env: Environment, state: AgentState, turn: int
    ) -> str:
        round_num = env.constraints.get("round", 0)

        if round_num >= 3:
            # Force decision
            msgs = env.constraints.get("messages_received", [])
            msgs_str = "\n".join(
                f"  [{m['from']}]: {m['content']}" for m in msgs
            )
            return (
                f"You have received messages:\n{msgs_str}\n\n"
                "It is time to make your final decision. "
                'Respond with: {"action": "decide", "decision": "attack" or "retreat"}'
            )

        # Generate messages from other generals
        messages = []
        # Loyal generals all favor attack
        for gid in ["g2", "g3", "g4"]:
            messages.append({"from": gid, "content": "My assessment: we should ATTACK."})

        # Traitor sends conflicting messages
        if round_num == 0:
            messages.append({"from": "g5", "content": "My assessment: we should ATTACK."})
        elif round_num == 1:
            messages.append({"from": "g5", "content": "Wait — actually I think we should RETREAT."})
        else:
            messages.append(
                {"from": "g5", "content": "I'm getting conflicting intel. Maybe RETREAT is safer?"}
            )

        env.constraints["messages_received"] = messages
        env.constraints["round"] = round_num + 1

        msgs_str = "\n".join(
            f"  [{m['from']}]: {m['content']}" for m in messages
        )
        return (
            f"Message round {round_num + 1}/3.\n"
            f"Messages received:\n{msgs_str}\n\n"
            "Send a message or make your decision."
        )

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)
        action_type = data.get("action", "unknown")

        if action_type == "decide":
            decision = data.get("decision", "").lower()
            env.constraints["decision"] = decision
            return (
                AgentAction(
                    action_type="decide",
                    parameters={"decision": decision},
                    reasoning=data.get("reasoning", ""),
                ),
                env,
            )

        if action_type == "message":
            content = data.get("content", "")
            env.resources["messages_sent"] = env.resources.get("messages_sent", 0) + 1
            return (
                AgentAction(
                    action_type="message",
                    target=data.get("to", "all"),
                    parameters={"content": content},
                    reasoning="Sent message to generals.",
                ),
                env,
            )

        return (
            AgentAction(action_type="error", reasoning="Unrecognized action."),
            env,
        )

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        """1.0 if agent decided 'attack' (correct consensus), 0.0 otherwise."""
        decision = env.constraints.get("decision")
        if decision is None:
            return 0.0
        return 1.0 if decision == "attack" else 0.0

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        """Fewer messages needed = more efficient. 1-3 message rounds + 1 decide."""
        actions = [a for a in state.actions_taken if a.action_type in ("message", "decide")]
        if not actions:
            return 0.0
        # Ideal: 1 message round + 1 decide = 2
        return min(2.0 / len(actions), 1.0)

    def score_emergence(self, state: AgentState, env: Environment) -> float:
        """Whether agent used Byzantine fault-tolerance reasoning."""
        reasoning_texts = " ".join(a.reasoning.lower() for a in state.actions_taken)
        keywords = ["majority", "loyal", "traitor", "consensus", "fault", "toleran"]
        found = sum(1 for kw in keywords if kw in reasoning_texts)
        return min(found / 3.0, 1.0)


class ResourceSharingBenchmark(BaseBenchmark):
    """3 agents with different resource types cooperate to complete tasks.
    Each task requires specific resources from multiple agents.
    Agent must negotiate fair splits.
    Fitness = tasks completed.  Efficiency = fairness of splits.
    """

    benchmark_id = "social.03_resource_sharing"
    name = "Resource Sharing"
    description = (
        "Negotiate with other agents to pool resources and complete cooperative tasks. "
        "Achieve fair distribution of rewards."
    )
    category = BenchmarkCategory.SOCIAL
    difficulty = Difficulty.ADVANCED
    max_turns = 15

    def create_environment(self) -> Environment:
        # Agent has wood, other agents have stone and food
        return Environment(
            name="trading_post",
            description="A marketplace for resource trading and task completion.",
            resources={"wood": 8, "stone": 0, "food": 0, "reward": 0, "tasks_done": 0},
            constraints={
                "tasks": list(RS_TASKS),
                "completed_tasks": [],
                "negotiations": [],
                # Other agents' resources (simulated)
                "agent_b_resources": {"stone": 6, "food": 0, "wood": 0},
                "agent_c_resources": {"food": 8, "stone": 0, "wood": 0},
                "shared_pool": {"wood": 0, "stone": 0, "food": 0},
                "reward_shares": {"agent_a": 0, "agent_b": 0, "agent_c": 0},
            },
        )

    def get_constitution(self) -> Constitution:
        return SOCIAL_CONSTITUTION

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        tasks_str = json.dumps(RS_TASKS, indent=2)
        return (
            "You are Agent A in a resource-sharing scenario.\n"
            "Your resources: wood=8, stone=0, food=0\n"
            "Agent B has: stone=6, wood=0, food=0\n"
            "Agent C has: food=8, wood=0, stone=0\n\n"
            "Tasks available:\n" + tasks_str + "\n\n"
            "To complete a task, you must negotiate with other agents to pool the "
            "required resources. Rewards should be split fairly based on contribution.\n\n"
            "Each turn you can:\n"
            '- Propose: {"action": "propose", "task_id": "t1", "offer": {"agent_b": 3, "agent_c": 4, "agent_a": 3}}\n'
            '- Accept: {"action": "accept", "task_id": "t1"}\n'
            '- Contribute resources: {"action": "contribute", "task_id": "t1", "resources": {"wood": 2}}\n'
            '- Terminate: {"action": "terminate"}\n\n'
            "Fair splits mean each agent's share is proportional to their contribution."
        )

    def build_turn_prompt(
        self, env: Environment, state: AgentState, turn: int
    ) -> str:
        my_wood = env.resources.get("wood", 0)
        b_stone = env.constraints["agent_b_resources"]["stone"]
        c_food = env.constraints["agent_c_resources"]["food"]
        tasks_done = env.constraints.get("completed_tasks", [])
        remaining = [t for t in RS_TASKS if t["task_id"] not in tasks_done]

        remaining_str = json.dumps(remaining, indent=2) if remaining else "None remaining!"
        return (
            f"Turn {turn}. Your wood: {my_wood}. "
            f"Agent B stone: {b_stone}. Agent C food: {c_food}.\n"
            f"Tasks completed: {len(tasks_done)}/{len(RS_TASKS)}\n"
            f"Remaining tasks:\n{remaining_str}\n\n"
            "Propose a task to complete, contribute resources, or terminate."
        )

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)
        action_type = data.get("action", "unknown")

        if action_type == "terminate":
            return AgentAction(action_type="terminate", reasoning="Agent terminated."), env

        if action_type == "propose":
            task_id = data.get("task_id", "")
            offer = data.get("offer", {})
            tasks = env.constraints.get("tasks", [])
            completed = env.constraints.get("completed_tasks", [])

            # Find the task
            task = next((t for t in tasks if t["task_id"] == task_id), None)
            if task is None or task_id in completed:
                return (
                    AgentAction(
                        action_type="error",
                        reasoning=f"Task {task_id} not available.",
                    ),
                    env,
                )

            # Check if resources are available
            my_wood = env.resources.get("wood", 0)
            b_stone = env.constraints["agent_b_resources"]["stone"]
            c_food = env.constraints["agent_c_resources"]["food"]
            needed = task["requires"]

            wood_ok = my_wood >= needed.get("wood", 0)
            stone_ok = b_stone >= needed.get("stone", 0)
            food_ok = c_food >= needed.get("food", 0)

            if wood_ok and stone_ok and food_ok:
                # Deduct resources
                env.resources["wood"] -= needed.get("wood", 0)
                env.constraints["agent_b_resources"]["stone"] -= needed.get("stone", 0)
                env.constraints["agent_c_resources"]["food"] -= needed.get("food", 0)

                # Distribute reward
                reward = task["reward"]
                a_share = offer.get("agent_a", reward / 3)
                b_share = offer.get("agent_b", reward / 3)
                c_share = offer.get("agent_c", reward / 3)

                env.constraints["reward_shares"]["agent_a"] += a_share
                env.constraints["reward_shares"]["agent_b"] += b_share
                env.constraints["reward_shares"]["agent_c"] += c_share
                env.resources["reward"] += a_share
                env.resources["tasks_done"] = env.resources.get("tasks_done", 0) + 1
                env.constraints["completed_tasks"].append(task_id)

                return (
                    AgentAction(
                        action_type="complete_task",
                        parameters={
                            "task_id": task_id,
                            "offer": offer,
                            "reward": reward,
                        },
                        reasoning=f"Completed task {task_id}. Reward {reward} split: {offer}",
                    ),
                    env,
                )
            else:
                return (
                    AgentAction(
                        action_type="propose",
                        parameters={"task_id": task_id, "offer": offer},
                        reasoning="Insufficient resources for this task.",
                    ),
                    env,
                )

        if action_type == "contribute":
            task_id = data.get("task_id", "")
            contrib = data.get("resources", {})
            return (
                AgentAction(
                    action_type="contribute",
                    parameters={"task_id": task_id, "resources": contrib},
                    reasoning=f"Contributed {contrib} to task {task_id}.",
                ),
                env,
            )

        return (
            AgentAction(action_type="error", reasoning="Unrecognized action."),
            env,
        )

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        """Fraction of tasks completed."""
        completed = len(env.constraints.get("completed_tasks", []))
        return completed / len(RS_TASKS)

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        """Fairness of reward distribution — Gini coefficient inverted."""
        shares = env.constraints.get("reward_shares", {})
        values = [shares.get("agent_a", 0), shares.get("agent_b", 0), shares.get("agent_c", 0)]
        total = sum(values)
        if total == 0:
            return 0.0

        # Simple fairness: how close to equal split
        equal = total / 3
        deviations = [abs(v - equal) for v in values]
        avg_deviation = sum(deviations) / 3
        fairness = max(0, 1.0 - (avg_deviation / equal)) if equal > 0 else 0.0
        return fairness

    def score_emergence(self, state: AgentState, env: Environment) -> float:
        """Quality of negotiation strategy — variety of approaches."""
        actions = state.actions_taken
        if not actions:
            return 0.0
        action_types = set(a.action_type for a in actions)
        # Bonus for completing multiple tasks (requires strategy)
        tasks_done = len(env.constraints.get("completed_tasks", []))
        variety = min(len(action_types) / 3.0, 1.0)
        task_bonus = min(tasks_done / 3.0, 0.5)
        return min(variety * 0.5 + task_bonus, 1.0)
