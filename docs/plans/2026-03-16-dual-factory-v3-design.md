# DESCARTES Dual Factory v3.0 — Implementation Design

*2026-03-16 | ARIA COGITO Programme*

---

## Overview

Implement the full Dual Factory v3.0 as described in `DESCARTES_DUAL_FACTORY_V3 LLM(1).md`. This adds 43 probe methods across 7 tiers, a 13-method statistical hardening suite, SAE superposition analysis, MLP nonlinear probing, a surrogate genome evolution system with LLM balloon expansion and DreamCoder pattern synthesis, and a unified zombie verdict generator — all integrated into the existing `L5PC/l5pc/` package.

## Guiding Principles

1. **Existing code untouched** — `ridge_probe.py`, `ablation.py`, `lstm.py`, `train.py` are not modified.
2. **Verbatim from guide** — Code is transcribed directly from the LLM(1) guide with minimal adaptation (imports, logging, integration).
3. **Registry-driven graceful degradation** — Optional deps (ripser, pysindy, cebra) checked at import time. One WARNING per session per missing package, then silent. `AVAILABLE_PROBES` dict gates the orchestrator.
4. **Core must never fail** — Ridge, MLP, SAE, statistical hardening, resample ablation depend only on PyTorch + scikit-learn + scipy.

## File Structure

```
l5pc/
  config.py                              # EXISTING — add SAE/MLP/hardening constants

  probing/
    __init__.py                          # EXISTING — add re-exports
    registry.py                          # NEW — AVAILABLE_PROBES, one-time warnings, probe interface
    ridge_probe.py                       # EXISTING — unchanged
    ablation.py                          # EXISTING — unchanged
    baselines.py                         # EXISTING — unchanged
    classify.py                          # EXISTING — unchanged
    mlp_probe.py                         # NEW — MLPProbe, mlp_delta_r2, _classify_encoding
    sae_probe.py                         # NEW — SparseAutoencoder, train_sae, sae_probe_biological_variables
    joint_alignment.py                   # NEW — CCA, RSA, CKA, pi-VAE, CEBRA, Procrustes
    dynamical_probes.py                  # NEW — Koopman, SINDy, DSA, iDSA, tangling, Lyapunov
    topological_probes.py                # NEW — TDA/persistent homology, manifold dimension
    causal_probes.py                     # NEW — DAS, transfer entropy, CCM
    information_probes.py                # NEW — MINE, MDL
    temporal_probes.py                   # NEW — temporal windows, gen matrices, gate-specific, adversarial

    hardening/                           # NEW — statistical hardening sub-package
      __init__.py                        # exports hardened_probe()
      permutation.py                     # block_permute, phase_randomize, circular_shift, adaptive_block_size
      diagnostics.py                     # durbin_watson, ljung_box, effective_dof
      corrections.py                     # fdr_correction, tost_zombie, bayes_factor_null
      frequency.py                       # frequency_resolved_r2, partial_coherence_r2
      gap_cv.py                          # gap_temporal_cv, cluster_permutation_test

  surrogates/
    __init__.py                          # EXISTING — add re-exports
    lstm.py                              # EXISTING — unchanged
    tcn.py                               # EXISTING — unchanged
    train.py                             # EXISTING — unchanged
    extract_hidden.py                    # EXISTING — unchanged
    surrogate_registry.py                # NEW — architecture builder dispatch from SurrogateGenome_v3

  factory/                               # NEW — entire sub-package
    __init__.py
    config.py                            # Factory-specific constants (Thompson priors, DreamCoder
                                         #   schedule, LLM balloon threshold, output CC gate,
                                         #   fitness weights alpha/beta/gamma, phase boundaries)
    probe_genome.py                      # ProbeGenome_v3 dataclass
    surrogate_genome.py                  # SurrogateGenome_v3, SurrogateGenomeComposer
    surrogate_trainer.py                 # SurrogateTrainer, output validation gate
    surrogate_fitness.py                 # SurrogateFitness multi-objective
    surrogate_factory.py                 # SurrogateFactory (outer loop, 4-phase campaign)
    probing_evaluator.py                 # ProbingFactoryEvaluator (inner loop, tiered with early termination)
    orchestrator.py                      # DualFactoryOrchestrator, Thompson sampling, DP clustering
    verdict.py                           # ZombieVerdictGenerator_v3
    llm_balloon.py                       # LLMBalloonExpander, system prompts for C1 and C2
    dreamcoder.py                        # SurrogateDreamCoder wake-sleep pattern library
```

## Registry Pattern (`probing/registry.py`)

```python
# At import time:
AVAILABLE_PROBES = {}
_WARNED = set()

def _check_optional(name, package, install_hint):
    try:
        __import__(package)
        AVAILABLE_PROBES[name] = True
    except ImportError:
        AVAILABLE_PROBES[name] = False
        if name not in _WARNED:
            logger.warning("WARNING: %s not installed — %s probes disabled (%s)",
                           package, name, install_hint)
            _WARNED.add(name)

# Core (always True)
AVAILABLE_PROBES['ridge'] = True
AVAILABLE_PROBES['mlp'] = True
AVAILABLE_PROBES['sae'] = True
AVAILABLE_PROBES['hardening'] = True
AVAILABLE_PROBES['resample_ablation'] = True

# Optional
_check_optional('tda', 'ripser', 'pip install ripser persim')
_check_optional('cebra', 'cebra', 'pip install cebra')
_check_optional('sindy', 'pysindy', 'pip install pysindy')
_check_optional('cca', 'sklearn.cross_decomposition', 'part of scikit-learn')  # Always available
```

The orchestrator checks `AVAILABLE_PROBES[probe_name]` before scheduling any probe.

## Statistical Hardening Sub-Package

`from l5pc.probing.hardening import hardened_probe` — single entry point.

| File | Contents | ~Lines |
|------|----------|--------|
| `permutation.py` | `block_permute`, `adaptive_block_size`, `phase_randomize` (IAAFT), `circular_shift_null` | ~120 |
| `diagnostics.py` | `durbin_watson`, `ljung_box_residual_test`, `effective_dof` | ~80 |
| `corrections.py` | `fdr_correction`, `tost_zombie_test`, `bayes_factor_null` | ~100 |
| `frequency.py` | `frequency_resolved_r2`, `partial_coherence_r2` | ~80 |
| `gap_cv.py` | `gap_temporal_cv`, `cluster_permutation_test` | ~100 |
| `__init__.py` | `hardened_probe()` orchestrator + `_hardened_verdict()` | ~120 |

Each file stays under 150 lines. Circuit-specific extensions (adaptive block size per circuit, custom frequency bands) go in the appropriate file without bloating others.

## Factory Config (`factory/config.py`)

Separate from `l5pc/config.py`. Contains:

```python
# Thompson sampling
THOMPSON_PRIOR_ALPHA = 1
THOMPSON_PRIOR_BETA = 1

# DreamCoder wake-sleep
DREAMCODER_WAKE_INTERVAL = 25      # rounds
DREAMCODER_MIN_PATTERN_EVALS = 5
DREAMCODER_PRUNE_THRESHOLD = 0.2

# LLM balloon
LLM_BALLOON_THRESHOLD = 50         # stale rounds before triggering
LLM_MODEL = 'claude-sonnet-4-20250514'
LLM_MAX_TOKENS = 2000

# Output validation gate
OUTPUT_CC_THRESHOLD = 0.7

# Fitness weights
FITNESS_ALPHA = 0.3                 # output accuracy
FITNESS_BETA = 0.5                  # biological correspondence
FITNESS_GAMMA = 0.2                 # causal necessity

# Campaign phases (fraction of total rounds)
PHASE_1_END = 0.25                  # template + random
PHASE_2_END = 0.50                  # mutation + crossover
PHASE_3_END = 0.75                  # DreamCoder library
# Phase 4: LLM balloon (0.75 to 1.0)

# Surrogate genome search space
ARCHITECTURE_OPTIONS = [
    'lstm', 'gru', 'neural_ode', 'ude', 'ltc', 'rmm',
    'mamba', 'neural_cde', 'koopman_ae', 'volterra',
    'tcn', 'transformer', 'pinn'
]
HIDDEN_DIM_OPTIONS = [16, 32, 64, 128, 256, 512]
BIO_LOSS_WEIGHT_OPTIONS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
```

## Config Additions to `l5pc/config.py`

```python
# === SAE (Section 3) ===
SAE_EXPANSION_FACTORS = [4, 8, 16]
SAE_K = 20
SAE_LR = 3e-4
SAE_EPOCHS = 50
SAE_BATCH_SIZE = 4096
SAE_DEAD_FEATURE_THRESHOLD = 0.80

# === MLP Probe (Section 4) ===
MLP_PROBE_HIDDEN_DIM = 64
MLP_PROBE_EPOCHS = 50
MLP_PROBE_LR = 1e-3
MLP_NONLINEAR_GAIN_THRESHOLD = 0.1

# === Statistical Hardening (Section 5) ===
HARDENING_N_BLOCK_PERMS = 500
HARDENING_N_PHASE_SURROGATES = 200
HARDENING_N_CIRCULAR_SHIFTS = 1000
HARDENING_MIN_BLOCK_SIZE = 50
HARDENING_IAAFT_ITERATIONS = 20
HARDENING_FDR_ALPHA = 0.05
HARDENING_TOST_EQUIVALENCE_BOUND = 0.05
HARDENING_BF_PRIOR_SCALE = 0.1
HARDENING_GAP_SIZE = 50

# === Frequency Bands (Section 5.6) ===
FREQUENCY_BANDS = {
    'ultra_slow': (0.1, 1),
    'slow': (1, 10),
    'medium': (10, 100),
    'fast': (100, 450),
}

# === Probe Tiers ===
CCA_N_COMPONENTS = 10
CCA_N_PERMUTATIONS = 200
RSA_N_SAMPLES = 2000
KOOPMAN_N_MODES = 20
DSA_DELAY = 1
DSA_RANK = 20
TDA_MAX_DIM = 2
TDA_N_LANDMARKS = 200
MINE_EPOCHS = 200
MINE_BATCH_SIZE = 512
MDL_N_PORTIONS = 10
TEMPORAL_WINDOW_SIZES = [1, 5, 10, 20, 50]
```

## Implementation Order

Matches the guide's 4-phase roadmap:

### Phase 1: HIGH PRIORITY
1. `probing/registry.py`
2. `probing/mlp_probe.py`
3. `probing/sae_probe.py`
4. `probing/hardening/` (all 6 files)
5. `config.py` additions

### Phase 2: CORE PROBES
6. `probing/joint_alignment.py`
7. `probing/causal_probes.py`
8. `probing/dynamical_probes.py`

### Phase 3: EXTENDED PROBES
9. `probing/topological_probes.py`
10. `probing/information_probes.py`
11. `probing/temporal_probes.py`

### Phase 4: INTEGRATION
12. `factory/config.py`
13. `factory/probe_genome.py`
14. `factory/surrogate_genome.py`
15. `factory/surrogate_trainer.py`
16. `factory/surrogate_fitness.py`
17. `factory/verdict.py`
18. `factory/llm_balloon.py`
19. `factory/dreamcoder.py`
20. `factory/probing_evaluator.py`
21. `factory/surrogate_factory.py`
22. `factory/orchestrator.py`
23. `surrogates/surrogate_registry.py`
24. `requirements-v3.txt`

## Requirements

**`requirements-v3.txt`** (optional deps for full suite):
```
# Core (already installed)
# torch, scikit-learn, scipy, numpy

# Tier 3: Dynamical
pysindy>=2.0

# Tier 4: Topological
ripser>=0.6
persim>=0.3

# Tier 2: Joint alignment (optional)
cebra>=0.4

# LLM balloon
anthropic>=0.40
```

## Source of Truth

All code is transcribed from `DESCARTES_DUAL_FACTORY_V3 LLM(1).md`. Sections 3-15 contain production-ready Python. The guide IS the spec.
