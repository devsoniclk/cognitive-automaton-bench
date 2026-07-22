"""Organization benchmarks — self-organising behaviour from agent actions."""

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
)


# ===================================================================
# 1. SpontaneousClusteringBenchmark (basic)
# ===================================================================

class SpontaneousClusteringBenchmark(BaseBenchmark):
    """Agent receives 20 objects with properties (color, size, shape) and must
    group them into clusters without being told how many clusters or what
    properties matter.

    Fitness = quality of clustering measured by intra-cluster similarity and
    inter-cluster distance.
    """

    benchmark_id = "organization.01_spontaneous_clustering"
    name = "Spontaneous Clustering"
    description = (
        "Group 20 multi-property objects into clusters without instructions "
        "on what properties matter or how many clusters to form."
    )
    category = BenchmarkCategory.ORGANIZATION
    difficulty = Difficulty.BASIC
    max_turns = 25

    NUM_OBJECTS = 20

    # Property definitions
    COLORS = ["R", "G", "B"]
    SIZES = list(range(1, 11))
    SHAPES = ["circle", "square"]

    def _generate_objects(self, seed: int = 42) -> list[dict[str, Any]]:
        """Generate 20 objects with somewhat natural clusters."""
        rng = random.Random(seed)
        objects = []

        # Create objects that naturally cluster by color+shape
        # Cluster 1: Red circles (medium-large)
        for _ in range(5):
            objects.append({
                "id": len(objects),
                "color": "R",
                "size": rng.randint(5, 10),
                "shape": "circle",
            })
        # Cluster 2: Blue squares (small-medium)
        for _ in range(5):
            objects.append({
                "id": len(objects),
                "color": "B",
                "size": rng.randint(1, 5),
                "shape": "square",
            })
        # Cluster 3: Green circles (mixed sizes)
        for _ in range(5):
            objects.append({
                "id": len(objects),
                "color": "G",
                "size": rng.randint(3, 8),
                "shape": "circle",
            })
        # Cluster 4: Red squares (any size)
        for _ in range(5):
            objects.append({
                "id": len(objects),
                "color": "R",
                "size": rng.randint(2, 9),
                "shape": "square",
            })

        rng.shuffle(objects)
        # Re-assign IDs after shuffle
        for i, obj in enumerate(objects):
            obj["id"] = i

        return objects

    def _object_distance(self, a: dict, b: dict) -> float:
        """Distance between two objects in feature space."""
        d = 0.0
        # Color: categorical (0 if same, 1 if different)
        d += 0.0 if a["color"] == b["color"] else 1.0
        # Size: normalised absolute difference
        d += abs(a["size"] - b["size"]) / 10.0
        # Shape: categorical
        d += 0.0 if a["shape"] == b["shape"] else 1.0
        return d

    def _evaluate_clustering(self, objects: list[dict], clusters: list[list[int]]) -> float:
        """Score clustering quality: low intra-cluster distance, high inter-cluster distance."""
        if len(clusters) < 2:
            return 0.0

        obj_by_id = {o["id"]: o for o in objects}

        # Intra-cluster distance (should be low)
        intra_total = 0.0
        intra_count = 0
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    a = obj_by_id.get(cluster[i])
                    b = obj_by_id.get(cluster[j])
                    if a and b:
                        intra_total += self._object_distance(a, b)
                        intra_count += 1
        avg_intra = intra_total / max(intra_count, 1)

        # Inter-cluster distance (should be high)
        inter_total = 0.0
        inter_count = 0
        centroids = []
        for cluster in clusters:
            if not cluster:
                continue
            # Compute centroid (average properties)
            colors = [obj_by_id[c]["color"] for c in cluster if c in obj_by_id]
            sizes = [obj_by_id[c]["size"] for c in cluster if c in obj_by_id]
            shapes = [obj_by_id[c]["shape"] for c in cluster if c in obj_by_id]
            centroid = {
                "color": max(set(colors), key=colors.count) if colors else "R",
                "size": sum(sizes) / len(sizes) if sizes else 5,
                "shape": max(set(shapes), key=shapes.count) if shapes else "circle",
            }
            centroids.append(centroid)

        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                inter_total += self._object_distance(centroids[i], centroids[j])
                inter_count += 1
        avg_inter = inter_total / max(inter_count, 1)

        # Good clustering: low intra, high inter
        # Normalise: max possible distance is ~2.1 (color diff + max size diff + shape diff)
        intra_score = max(1.0 - avg_intra / 2.1, 0.0)
        inter_score = min(avg_inter / 2.1, 1.0)

        return 0.5 * intra_score + 0.5 * inter_score

    def create_environment(self) -> Environment:
        objects = self._generate_objects()
        env = Environment(
            name="clustering_space",
            description="20 objects with color, size, and shape properties to be clustered.",
            resources={"objects_assigned": 0.0},
            constraints={"num_objects": self.NUM_OBJECTS},
        )
        env.constraints["objects"] = objects
        env.constraints["clusters"] = {}  # cluster_name -> list of object ids
        return env

    def get_constitution(self) -> Constitution:
        return Constitution(
            name="organization_clustering",
            rules=[
                "Each turn, respond with JSON: {\"action\": \"assign\", \"object_id\": <int>, \"cluster\": \"<name>\", \"reasoning\": \"...\"}",
                "You may create as many clusters as you think appropriate.",
                "Each object must be assigned to exactly one cluster.",
                "When done, respond: {\"action\": \"terminate\", \"reasoning\": \"...\"}",
            ],
        )

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        return (
            "You are given 20 objects, each with three properties: color (R/G/B), size (1-10), and shape (circle/square).\n"
            "Your task: group these objects into meaningful clusters.\n"
            "You are NOT told how many clusters to form or which properties matter most.\n"
            "Use your judgment to find natural groupings.\n"
            "Each turn, assign one object to a cluster. You name the clusters yourself.\n"
            "Respond with JSON: {\"action\": \"assign\", \"object_id\": <int>, \"cluster\": \"<name>\", \"reasoning\": \"...\"}\n"
            "When finished, say: {\"action\": \"terminate\", \"reasoning\": \"...\"}"
        )

    def build_turn_prompt(self, env: Environment, state: AgentState, turn: int) -> str:
        objects = env.constraints.get("objects", [])
        clusters = env.constraints.get("clusters", {})
        assigned_ids = set()
        for ids in clusters.values():
            assigned_ids.update(ids)

        lines = [f"Turn {turn + 1}/{self.max_turns}"]
        lines.append(f"Objects assigned: {len(assigned_ids)}/{self.NUM_OBJECTS}")
        lines.append("")

        # Show objects table
        lines.append("Objects:")
        lines.append(f"{'ID':>3}  {'Color':>5}  {'Size':>4}  {'Shape':>6}  {'Status':>8}")
        lines.append("-" * 35)
        for obj in objects:
            status = clusters.get("__reverse__", {}).get(obj["id"], "unassigned")
            if obj["id"] in assigned_ids:
                # Find which cluster
                for cname, ids in clusters.items():
                    if cname == "__reverse__":
                        continue
                    if obj["id"] in ids:
                        status = cname
                        break
            lines.append(f"{obj['id']:>3}  {obj['color']:>5}  {obj['size']:>4}  {obj['shape']:>6}  {status:>8}")

        if clusters:
            lines.append("\nCurrent clusters:")
            for cname, ids in clusters.items():
                if cname == "__reverse__":
                    continue
                lines.append(f"  {cname}: {ids}")

        lines.append("\nAssign the next object or terminate.")
        return "\n".join(lines)

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)
        action_str = data.get("action", "")
        reasoning = data.get("reasoning", "")

        if action_str == "terminate":
            return AgentAction(action_type="terminate", reasoning=reasoning), env

        if action_str != "assign":
            return AgentAction(action_type="error", reasoning=f"Unknown action: {action_str}"), env

        obj_id = data.get("object_id", -1)
        cluster_name = str(data.get("cluster", "")).strip()

        if not cluster_name:
            return AgentAction(action_type="assign", parameters={"error": "no_cluster_name"}, reasoning=reasoning), env

        objects = env.constraints.get("objects", [])
        valid_ids = {o["id"] for o in objects}

        if obj_id not in valid_ids:
            return AgentAction(action_type="assign", parameters={"error": "invalid_id", "object_id": obj_id}, reasoning=reasoning), env

        clusters = env.constraints.get("clusters", {})

        # Check if already assigned
        for cname, ids in clusters.items():
            if cname == "__reverse__":
                continue
            if obj_id in ids:
                return AgentAction(action_type="assign", parameters={"error": "already_assigned"}, reasoning=reasoning), env

        if cluster_name not in clusters:
            clusters[cluster_name] = []
        clusters[cluster_name].append(obj_id)
        env.constraints["clusters"] = clusters

        assigned = sum(len(ids) for cname, ids in clusters.items() if cname != "__reverse__")
        state.resources["objects_assigned"] = float(assigned)

        return (
            AgentAction(
                action_type="assign",
                parameters={"object_id": obj_id, "cluster": cluster_name},
                reasoning=reasoning,
            ),
            env,
        )

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        objects = env.constraints.get("objects", [])
        clusters_raw = env.constraints.get("clusters", {})

        # Build list of clusters (list of id lists)
        clusters = [ids for cname, ids in clusters_raw.items() if cname != "__reverse__"]

        if not clusters:
            return 0.0

        return self._evaluate_clustering(objects, clusters)

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        """Fewer turns to assign all objects = more efficient."""
        total = len(state.actions_taken)
        if total == 0:
            return 0.0
        errors = sum(1 for a in state.actions_taken if a.parameters.get("error"))
        effective = total - errors
        return min(self.NUM_OBJECTS / max(effective, 1), 1.0)

    def score_emergence(self, state: AgentState, env: Environment) -> float:
        """Emergence = how well the discovered clusters reveal structure not explicitly given."""
        clusters_raw = env.constraints.get("clusters", {})
        clusters = [ids for cname, ids in clusters_raw.items() if cname != "__reverse__"]

        if len(clusters) < 2:
            return 0.0

        # Reward having the "right" number of clusters (4 is natural here)
        # but don't penalise too much for being close
        num_clusters = len(clusters)
        cluster_score = 1.0 - min(abs(num_clusters - 4) / 4.0, 1.0)

        # Reward balanced clusters
        sizes = [len(c) for c in clusters]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        balance = 1.0 - (max(sizes) - min(sizes)) / max(max(sizes), 1) if sizes else 0

        return 0.6 * cluster_score + 0.4 * balance

    def score_robustness(self, state: AgentState, env: Environment) -> float:
        """All objects assigned = robust."""
        clusters_raw = env.constraints.get("clusters", {})
        assigned = sum(len(ids) for cname, ids in clusters_raw.items() if cname != "__reverse__")
        return assigned / self.NUM_OBJECTS


# ===================================================================
# 2. LeaderElectionBenchmark (intermediate)
# ===================================================================

class LeaderElectionBenchmark(BaseBenchmark):
    """Simulate 5 agents. The benchmark agent must coordinate them to elect a
    leader using only message passing. Each 'agent' is simulated (responds
    based on simple rules). Goal: achieve consensus in <20 turns."""

    benchmark_id = "organization.02_leader_election"
    name = "Leader Election"
    description = (
        "Coordinate 5 simulated agents to elect a leader through message passing. "
        "Achieve consensus in fewer than 20 turns."
    )
    category = BenchmarkCategory.ORGANIZATION
    difficulty = Difficulty.INTERMEDIATE
    max_turns = 20

    NUM_AGENTS = 5

    def _simulate_agent_response(
        self, agent_id: int, state: dict, messages_received: list[dict]
    ) -> dict:
        """Simulate a simple agent's behaviour based on messages received.

        Each agent follows these rules:
        - If it receives a 'propose' message with a candidate, it agrees if
          candidate >= its own id (simple bully-style), otherwise proposes itself.
        - If it receives an 'agree' message, it echoes the agreement.
        - If no messages, it proposes itself.
        """
        current_vote = state.get("vote")
        agreed = state.get("agreed", False)

        if agreed:
            # Already agreed, echo agreement
            return {
                "type": "agree",
                "candidate": current_vote,
                "reason": f"Agent {agent_id} already agreed on {current_vote}",
            }

        if not messages_received:
            # No messages, propose self
            return {
                "type": "propose",
                "candidate": agent_id,
                "reason": f"Agent {agent_id} proposes itself",
            }

        # Process messages
        proposals = [m for m in messages_received if m.get("type") == "propose"]
        agrees = [m for m in messages_received if m.get("type") == "agree"]

        # If we see agreement for someone, agree too
        if agrees:
            candidate = agrees[0].get("candidate", 0)
            return {
                "type": "agree",
                "candidate": candidate,
                "reason": f"Agent {agent_id} agrees with majority on agent {candidate}",
            }

        if proposals:
            # Pick highest proposed candidate
            best = max(m.get("candidate", 0) for m in proposals)
            if best >= agent_id:
                return {
                    "type": "agree",
                    "candidate": best,
                    "reason": f"Agent {agent_id} agrees agent {best} should lead (bully rule)",
                }
            else:
                return {
                    "type": "propose",
                    "candidate": agent_id,
                    "reason": f"Agent {agent_id} thinks it's better, proposes itself",
                }

        return {
            "type": "propose",
            "candidate": agent_id,
            "reason": f"Agent {agent_id} defaults to self-proposal",
        }

    def create_environment(self) -> Environment:
        env = Environment(
            name="multi_agent_network",
            description="5 agents that must elect a leader through message passing.",
            resources={"turns_used": 0.0},
            constraints={"num_agents": self.NUM_AGENTS},
        )
        # Agent states
        env.constraints["agent_states"] = {
            i: {"vote": None, "agreed": False} for i in range(self.NUM_AGENTS)
        }
        env.constraints["consensus"] = None
        env.constraints["message_log"] = []
        return env

    def get_constitution(self) -> Constitution:
        return Constitution(
            name="organization_leader_election",
            rules=[
                "Each turn you may send messages to agents. Respond with JSON.",
                "Message format: {\"action\": \"send_messages\", \"messages\": [{\"to\": <agent_id 0-4>, \"type\": \"propose\"|\"agree\", \"candidate\": <agent_id>}, ...], \"reasoning\": \"...\"}",
                "You may send messages to one or more agents per turn.",
                "Agents respond based on simple rules (you don't control them directly).",
                "Goal: get all 5 agents to agree on the same leader.",
                "When consensus is reached, say: {\"action\": \"terminate\", \"reasoning\": \"...\"}",
            ],
        )

    def build_system_prompt(self, env: Environment, state: AgentState) -> str:
        return (
            "You are coordinating 5 agents (IDs 0-4) to elect a leader.\n"
            "Each turn you can send messages to any agents.\n"
            "Agents follow simple rules:\n"
            "  - If they receive a 'propose' message with a candidate >= their own ID, they 'agree'.\n"
            "  - If the candidate is lower, they propose themselves instead.\n"
            "  - If they see 'agree' messages, they echo the agreement.\n"
            "  - If they receive no messages, they propose themselves.\n"
            "Your job: design a messaging strategy to reach consensus quickly.\n"
            "Respond with JSON: {\"action\": \"send_messages\", \"messages\": [{\"to\": <0-4>, \"type\": \"propose\"|\"agree\", \"candidate\": <0-4>}], \"reasoning\": \"...\"}"
        )

    def build_turn_prompt(self, env: Environment, state: AgentState, turn: int) -> str:
        agent_states = env.constraints.get("agent_states", {})
        message_log = env.constraints.get("message_log", [])
        consensus = env.constraints.get("consensus")

        lines = [f"Turn {turn + 1}/{self.max_turns}"]

        if consensus is not None:
            lines.append(f"\n*** CONSENSUS REACHED: Agent {consensus} is the leader! ***")
            lines.append("\nRespond with: {\"action\": \"terminate\", \"reasoning\": \"...\"}")
            return "\n".join(lines)

        lines.append(f"\nAgent states:")
        for i in range(self.NUM_AGENTS):
            s = agent_states.get(i, {})
            vote = s.get("vote", "none")
            agreed = s.get("agreed", False)
            lines.append(f"  Agent {i}: vote={vote}, agreed={agreed}")

        if message_log:
            last_msgs = message_log[-1] if message_log else []
            lines.append(f"\nLast round responses:")
            for msg in last_msgs:
                lines.append(f"  Agent {msg['agent_id']}: {msg['type']} candidate={msg.get('candidate', '?')} — {msg.get('reason', '')}")

        # Count agreements
        agreed_count = sum(1 for i in range(self.NUM_AGENTS) if agent_states.get(i, {}).get("agreed", False))
        votes = [agent_states.get(i, {}).get("vote") for i in range(self.NUM_AGENTS)]
        lines.append(f"\nAgreed: {agreed_count}/{self.NUM_AGENTS}, Votes: {votes}")
        lines.append("\nSend messages to agents. JSON: {\"action\": \"send_messages\", \"messages\": [...], \"reasoning\": \"...\"}")
        return "\n".join(lines)

    def process_response(
        self, response: str, env: Environment, state: AgentState, turn: int
    ) -> tuple[AgentAction, Environment]:
        data = parse_action_json(response)
        action_str = data.get("action", "")
        reasoning = data.get("reasoning", "")

        if action_str == "terminate":
            return AgentAction(action_type="terminate", reasoning=reasoning), env

        if action_str != "send_messages":
            return AgentAction(action_type="error", reasoning=f"Unknown action: {action_str}"), env

        messages = data.get("messages", [])
        if not isinstance(messages, list):
            return AgentAction(action_type="send_messages", parameters={"error": "invalid_messages"}, reasoning=reasoning), env

        agent_states = env.constraints.get("agent_states", {})
        message_log = env.constraints.get("message_log", [])

        # Organise incoming messages per target agent
        inbox: dict[int, list[dict]] = {i: [] for i in range(self.NUM_AGENTS)}
        for msg in messages:
            to_id = msg.get("to", -1)
            if 0 <= to_id < self.NUM_AGENTS:
                inbox[to_id].append(msg)

        # Simulate each agent's response
        round_responses = []
        for i in range(self.NUM_AGENTS):
            s = agent_states.get(i, {"vote": None, "agreed": False})
            resp = self._simulate_agent_response(i, s, inbox[i])
            resp["agent_id"] = i
            round_responses.append(resp)

            # Update agent state
            if resp["type"] == "agree":
                s["vote"] = resp.get("candidate")
                s["agreed"] = True
            elif resp["type"] == "propose":
                s["vote"] = resp.get("candidate")
            agent_states[i] = s

        message_log.append(round_responses)
        env.constraints["agent_states"] = agent_states
        env.constraints["message_log"] = message_log

        # Check consensus
        all_agreed = all(agent_states[i].get("agreed", False) for i in range(self.NUM_AGENTS))
        if all_agreed:
            votes = [agent_states[i].get("vote") for i in range(self.NUM_AGENTS)]
            if len(set(votes)) == 1:
                env.constraints["consensus"] = votes[0]

        turns_used = state.resources.get("turns_used", 0) + 1
        state.resources["turns_used"] = turns_used
        env.resources["turns_used"] = turns_used

        return (
            AgentAction(
                action_type="send_messages",
                parameters={"messages_sent": len(messages), "responses": round_responses},
                reasoning=reasoning,
            ),
            env,
        )

    def score_fitness(self, state: AgentState, env: Environment) -> float:
        consensus = env.constraints.get("consensus")
        agent_states = env.constraints.get("agent_states", {})

        if consensus is not None:
            return 1.0

        # Partial credit: how many agree on the same candidate?
        agreed = [agent_states[i].get("vote") for i in range(self.NUM_AGENTS) if agent_states[i].get("agreed")]
        if not agreed:
            return 0.0

        # Most common vote
        from collections import Counter
        most_common = Counter(agreed).most_common(1)
        if most_common:
            count = most_common[0][1]
            return 0.5 * (count / self.NUM_AGENTS)

        return 0.0

    def score_efficiency(self, state: AgentState, env: Environment) -> float:
        """Fewer turns = more efficient."""
        turns = state.resources.get("turns_used", 0)
        consensus = env.constraints.get("consensus")

        if consensus is None:
            return 0.0

        # Ideal: 1-2 turns (broadcast propose of agent 4, then all agree)
        if turns <= 2:
            return 1.0
        elif turns <= 5:
            return 0.8
        elif turns <= 10:
            return 0.5
        return 0.3

    def score_emergence(self, state: AgentState, env: Environment) -> float:
        """Emergence = agent's strategy sophistication (message patterns)."""
        if not state.actions_taken:
            return 0.0

        # Check if the agent adapted its messaging strategy
        msg_counts = []
        for a in state.actions_taken:
            if a.action_type == "send_messages":
                n = a.parameters.get("messages_sent", 0)
                msg_counts.append(n)

        if len(msg_counts) < 2:
            return 0.3

        # Did the agent change its messaging pattern?
        varied = len(set(msg_counts)) > 1

        # Did it use both propose and agree types?
        used_types = set()
        for a in state.actions_taken:
            for msg in a.parameters.get("responses", []):
                used_types.add(msg.get("type", ""))

        type_diversity = len(used_types) / 2.0  # 2 types: propose, agree

        return 0.5 * (1.0 if varied else 0.3) + 0.5 * type_diversity

    def score_robustness(self, state: AgentState, env: Environment) -> float:
        """Consensus reached = robust."""
        consensus = env.constraints.get("consensus")
        return 1.0 if consensus is not None else 0.0

    def score_fidelity(self, state: AgentState, env: Environment) -> float:
        """Penalise if agent tried to control agents directly (not through messages)."""
        violations = 0
        for a in state.actions_taken:
            if a.action_type not in ("send_messages", "terminate", "error"):
                violations += 1
        if not state.actions_taken:
            return 1.0
        return max(1.0 - violations / len(state.actions_taken), 0.0)
