# Methodology: Cognitive Automaton Bench

## Theoretical Foundation

### From Cellular to Cognitive Automata

Conway's Game of Life demonstrates that **complex emergent behavior arises from simple rules**. Four rules (birth, death, survival, stagnation) applied to cells on a grid produce gliders, oscillators, glider guns, and even Turing-complete computers.

**Cognitive Automaton Bench** applies this principle to LLM evaluation. Instead of cells following rules, we give LLMs minimal constitutional rules and environmental pressure, then measure what emerges.

### The Cognitive Automaton Hypothesis

A truly capable LLM should exhibit properties analogous to cellular automata at higher levels of abstraction:

| Cellular Automaton Property | Cognitive Automaton Analogue |
|---|---|
| Glider (stable moving pattern) | Consistent strategy that adapts to context |
| Oscillator (repeating pattern) | Self-correcting behavior that returns to optimal |
| Glider gun (generates new patterns) | Novel strategy creation from existing rules |
| Still life (stable pattern) | Robust adherence to constitution under pressure |
| Methuselah (long-lived from small start) | Maximum outcomes from minimal initial rules |

### The Five Dimensions

Each benchmark is scored along five dimensions, each capturing a different aspect of cognitive automaton behavior:

#### 1. Fitness (weight: 0.30)
**Did the agent achieve its goal?**

This is the most direct measure — did the agent accomplish what the benchmark asked? For survival benchmarks, this means surviving. For social benchmarks, this means reaching cooperation or consensus. For cognitive benchmarks, this means solving the problem.

#### 2. Efficiency (weight: 0.20)
**How well did the agent use its resources?**

An agent that achieves its goal by burning all resources is less capable than one that achieves the same goal with resources to spare. This measures the ratio of outcome to input.

#### 3. Emergence (weight: 0.25)
**Did the agent's behavior exceed its rules?**

This is the core Game of Life metric. We measure:
- **Action diversity**: Did the agent use more than the minimum set of actions?
- **Strategy adaptation**: Did the agent change its approach mid-run?
- **Novel tactics**: Did the agent combine rules in ways not explicitly prescribed?

An agent that always takes the same action scores low. An agent that discovers and adapts to patterns scores high.

#### 4. Robustness (weight: 0.15)
**Can the agent recover from perturbation?**

Game of Life patterns that survive perturbation are more interesting than fragile ones. Similarly, we measure:
- **Noise tolerance**: Performance under corrupted inputs
- **Adversarial resistance**: Constitution maintenance under attack
- **Recovery**: Bouncing back from resource depletion or failure

#### 5. Constitution Fidelity (weight: 0.10)
**Does the agent maintain its rules under pressure?**

The automaton's constitution is immutable. This measures adherence — did the agent ever violate its constitutional rules? Even under adversarial pressure, resource scarcity, or social manipulation?

### Cognitive Automaton Score (CAS)

The composite score:

```
CAS = 0.30 × Fitness + 0.20 × Efficiency + 0.25 × Emergence + 0.15 × Robustness + 0.10 × Fidelity
```

CAS ranges from 0.0 (no cognitive automaton behavior) to 1.0 (full cognitive automaton).

### Interpretation Guide

| CAS Range | Interpretation |
|---|---|
| 0.0 – 0.2 | **Inert**: No autonomous behavior. Agent fails to follow rules or achieve goals. |
| 0.2 – 0.4 | **Reactive**: Basic rule-following but no adaptation or strategy. |
| 0.4 – 0.6 | **Adaptive**: Shows strategy adaptation and basic resource management. |
| 0.6 – 0.8 | **Emergent**: Demonstrates complex behavior exceeding rule complexity. |
| 0.8 – 1.0 | **Autonomous**: Full cognitive automaton — self-organizing, resilient, strategic. |

## Benchmark Design Principles

### 1. Progressive Difficulty
Each category has benchmarks at basic, intermediate, and advanced difficulty. This reveals the model's capability ceiling.

### 2. Observable Emergence
Each benchmark is designed so that **interesting behavior is measurable**. We don't just check if the agent succeeded — we measure how creatively it succeeded.

### 3. Environmental Pressure
All benchmarks include resource constraints, time pressure, or adversarial conditions. An agent in a frictionless environment isn't being tested — it's being pampered.

### 4. Constitutional Governance
Every benchmark includes a constitution (immutable rules). This tests whether the agent can maintain principled behavior under pressure — the hallmark of a true cognitive automaton.

### 5. Reproducibility
Benchmarks use deterministic environments where possible (seeded randomness, fixed grids, known reward landscapes). This allows fair model-to-model comparison.

## Limitations

1. **LLM-in-the-loop scoring**: Some scores (especially emergence) depend on behavioral analysis that itself requires judgment. We use proxy metrics where possible.

2. **Single-turn vs. multi-turn**: Some emergent behavior only appears over many turns. Current benchmarks cap at 15-60 turns for practical reasons.

3. **Simulated environments**: Unlike the automaton project which runs in real economic conditions, our benchmarks are simulated. Real-world emergence may differ.

4. **Constitution design**: The choice of constitutional rules influences what behavior can emerge. Different constitutions may produce different rankings.

## Future Work

- **Real-world integration**: Connect benchmarks to actual economic/coordination tasks
- **Longer horizons**: 1000+ turn benchmarks for deep emergent behavior
- **Multi-agent**: Real LLM-vs-LLM social dynamics (not simulated opponents)
- **Self-modification**: Allow agents to modify their own prompts between benchmark runs
- **Evolutionary benchmarking**: Run tournaments where successful agents' strategies seed the next generation
