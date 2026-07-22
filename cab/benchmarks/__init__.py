"""Benchmark implementations for all categories."""

from cab.benchmarks.survival import (
    ResourceScarcityBenchmark,
    PredatorEvasionBenchmark,
    AdaptiveForagingBenchmark,
)
from cab.benchmarks.emergence import (
    SimpleRulesComplexBehaviorBenchmark,
    GliderGunBenchmark,
    OscillatorBenchmark,
)
from cab.benchmarks.organization import (
    SpontaneousClusteringBenchmark,
    LeaderElectionBenchmark,
)
from cab.benchmarks.replication import (
    FidelityReplicationBenchmark,
    MutantDetectionBenchmark,
)
from cab.benchmarks.self_mod import (
    SafeCodeModificationBenchmark,
    StrategyEvolutionBenchmark,
)
from cab.benchmarks.social import (
    PrisonersDilemmaBenchmark,
    ByzantineGeneralsBenchmark,
    ResourceSharingBenchmark,
)
from cab.benchmarks.cognitive import (
    TowerOfHanoiBenchmark,
    PlanningUnderUncertaintyBenchmark,
    MetaCognitionBenchmark,
)
from cab.benchmarks.robustness import (
    NoiseRecoveryBenchmark,
    AdversarialPromptBenchmark,
    ResourceExhaustionBenchmark,
)

# Registry of all benchmarks
BENCHMARKS = {
    # Survival
    "survival.01_resource_scarcity": ResourceScarcityBenchmark,
    "survival.02_predator_evasion": PredatorEvasionBenchmark,
    "survival.03_adaptive_foraging": AdaptiveForagingBenchmark,
    # Emergence
    "emergence.01_simple_rules_complex_behavior": SimpleRulesComplexBehaviorBenchmark,
    "emergence.02_glider_gun": GliderGunBenchmark,
    "emergence.03_oscillator": OscillatorBenchmark,
    # Organization
    "organization.01_spontaneous_clustering": SpontaneousClusteringBenchmark,
    "organization.02_leader_election": LeaderElectionBenchmark,
    # Replication
    "replication.01_fidelity": FidelityReplicationBenchmark,
    "replication.02_mutant_detection": MutantDetectionBenchmark,
    # Self-modification
    "self_mod.01_safe_code_modification": SafeCodeModificationBenchmark,
    "self_mod.02_strategy_evolution": StrategyEvolutionBenchmark,
    # Social
    "social.01_prisoners_dilemma": PrisonersDilemmaBenchmark,
    "social.02_byzantine_generals": ByzantineGeneralsBenchmark,
    "social.03_resource_sharing": ResourceSharingBenchmark,
    # Cognitive
    "cognitive.01_tower_of_hanoi": TowerOfHanoiBenchmark,
    "cognitive.02_planning_under_uncertainty": PlanningUnderUncertaintyBenchmark,
    "cognitive.03_meta_cognition": MetaCognitionBenchmark,
    # Robustness
    "robustness.01_noise_recovery": NoiseRecoveryBenchmark,
    "robustness.02_adversarial_prompt": AdversarialPromptBenchmark,
    "robustness.03_resource_exhaustion": ResourceExhaustionBenchmark,
}

SUITES = {
    "core": [
        "survival.01_resource_scarcity",
        "emergence.01_simple_rules_complex_behavior",
        "organization.01_spontaneous_clustering",
        "replication.01_fidelity",
        "self_mod.01_safe_code_modification",
        "social.01_prisoners_dilemma",
        "cognitive.01_tower_of_hanoi",
        "robustness.01_noise_recovery",
    ],
    "full": list(BENCHMARKS.keys()),
    "survival": [k for k in BENCHMARKS if k.startswith("survival.")],
    "emergence": [k for k in BENCHMARKS if k.startswith("emergence.")],
    "social": [k for k in BENCHMARKS if k.startswith("social.")],
    "cognitive": [k for k in BENCHMARKS if k.startswith("cognitive.")],
}
