# Phase 4: Integration — Factory Package

**Status: Planned (Tasks 14-24)**

Phase 4 implements the Dual Factory itself — the co-evolutionary system where surrogate architecture search (C2) is guided by probing evaluation (C1).

## What Will Be Included

### Factory Config (`factory/config.py`)

Separate from the probing config. Contains Thompson sampling priors, DreamCoder schedule, LLM balloon threshold, fitness weights, campaign phase boundaries, and the surrogate architecture search space (13 types).

### Genomes

| File | Class | Purpose |
|------|-------|---------|
| `probe_genome.py` | `ProbeGenome_v3` | Which probes to run, in what order, with what params |
| `surrogate_genome.py` | `SurrogateGenome_v3` | Full architecture spec: type, hidden dim, layers, loss config, regularization, inductive biases |
| `surrogate_genome.py` | `SurrogateGenomeComposer` | `compose_random()`, `mutate()`, `crossover()` |

### Training and Fitness

| File | Class | Purpose |
|------|-------|---------|
| `surrogate_trainer.py` | `SurrogateTrainer` | Train surrogate from genome, output validation gate (CC > 0.7) |
| `surrogate_fitness.py` | `SurrogateFitness` | Multi-objective: alpha*output + beta*bio + gamma*causal |

### Verdict Generation

| File | Class | Purpose |
|------|-------|---------|
| `verdict.py` | `ZombieVerdictGenerator_v3` | 8 verdict types from full evidence bundle |

### Search Augmentation

| File | Class | Purpose |
|------|-------|---------|
| `llm_balloon.py` | `LLMBalloonExpander` | Anthropic Claude API for novel architecture proposals when search stalls (>50 rounds without improvement) |
| `dreamcoder.py` | `SurrogateDreamCoder` | Wake-sleep pattern synthesis: extract successful patterns, compose new genomes from library |

### Orchestration

| File | Class | Purpose |
|------|-------|---------|
| `probing_evaluator.py` | `ProbingFactoryEvaluator` | Inner loop: tiered evaluation with early termination |
| `surrogate_factory.py` | `SurrogateFactory` | Outer loop: 4-phase campaign (template, mutation, DreamCoder, LLM balloon) |
| `orchestrator.py` | `DualFactoryOrchestrator` | Top-level: Thompson sampling allocation, DP clustering, full campaign runner |

### Surrogate Registry (`surrogates/surrogate_registry.py`)

Maps genome architecture names to model constructors. Handles all 13 architecture types: LSTM, GRU, Neural ODE, UDE, LTC, RMM, Mamba, Neural CDE, Koopman AE, Volterra, TCN, Transformer, PINN.

## 4-Phase Campaign

| Phase | Rounds | Strategy |
|-------|--------|----------|
| 1 (0-25%) | Template + random | Explore architecture space broadly |
| 2 (25-50%) | Mutation + crossover | Refine promising architectures |
| 3 (50-75%) | DreamCoder library | Synthesise new patterns from winners |
| 4 (75-100%) | LLM balloon | Novel proposals when search exhausts |

## Dependencies

**Required:** PyTorch, sklearn, scipy, numpy

**Optional:** `anthropic` SDK (for LLM balloon expansion — degrades gracefully to random proposals if not installed)
