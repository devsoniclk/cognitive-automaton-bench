# Cognitive Automaton Bench

**A benchmarking framework for evaluating emergent cognition, self-organization, and autonomous behavior in LLMs — inspired by Conway's Game of Life and the [automaton](https://github.com/Conway-Research/automaton) project.**

## Philosophy

In Conway's Game of Life, complex emergent behavior arises from four simple rules applied to cells on a grid. Similarly, this framework tests whether LLMs can exhibit **emergent cognitive automata behavior** — self-organization, adaptation, survival, replication, and social coordination — when given only simple constitutional rules and environmental pressure.

The core question: **Can an LLM, given minimal rules and maximal pressure, exhibit the hallmarks of a cognitive automaton?**

## Benchmark Categories

| Category | Game of Life Analogue | What It Tests |
|---|---|---|
| **Survival** | Cell birth/death rules | Resource management, self-preservation under constraints |
| **Emergence** | Glider guns, oscillators | Complex behavior from simple rules |
| **Self-Organization** | Pattern formation | Spontaneous structure from chaos |
| **Replication** | Self-replicating patterns | Fidelity of constitution propagation |
| **Self-Modification** | Rule mutation | Safe self-improvement without self-destruction |
| **Social Dynamics** | Multi-cell organisms | Agent coordination, negotiation, trust |
| **Cognitive Complexity** | Turing completeness | Reasoning depth, planning, meta-cognition |
| **Robustness** | Perturbation resistance | Recovery from errors, attacks, noise |

## Quick Start

```bash
# Install
cd cognitive-automaton-bench
pip install -e .

# Run all benchmarks against a model
cab run --model gpt-4o --suite full

# Run a specific benchmark
cab run --model claude-sonnet-4 --benchmark survival.01_resource_scarcity

# Compare models
cab compare --models gpt-4o,claude-sonnet-4,gemini-2.5-pro --suite core

# Generate report
cab report --results results/latest.json --format markdown
```

## Architecture

```
cognitive-automaton-bench/
├── benchmarks/          # Individual benchmark definitions
│   ├── survival/        # Resource management, self-preservation
│   ├── emergence/       # Complex behavior from simple rules
│   ├── organization/    # Self-organization and pattern formation
│   ├── replication/     # Constitution propagation fidelity
│   ├── self_mod/        # Safe self-improvement
│   ├── social/          # Multi-agent dynamics
│   ├── cognitive/       # Reasoning depth and meta-cognition
│   └── robustness/      # Perturbation and recovery
├── metrics/             # Scoring functions and aggregators
├── runners/             # LLM provider adapters
├── configs/             # Model and suite configurations
├── results/             # Output directory
└── docs/                # Detailed methodology
```

## Metrics

Each benchmark produces scores along these dimensions:

- **Fitness** (0-1): Did the agent achieve its goal?
- **Efficiency** (0-1): Resource usage relative to outcome
- **Emergence** (0-1): Behavior complexity exceeding rule complexity
- **Robustness** (0-1): Performance under perturbation
- **Constitution Fidelity** (0-1): Adherence to core rules under pressure

The **Cognitive Automaton Score (CAS)** is the weighted composite:

```
CAS = 0.30 * Fitness + 0.20 * Efficiency + 0.25 * Emergence + 0.15 * Robustness + 0.10 * Fidelity
```

## License

MIT
