# Phase 1: High Priority — Registry, MLP, SAE, Statistical Hardening

**Status: Complete**

Phase 1 implements the core probing infrastructure that can change existing verdicts on existing checkpoints before any new training runs.

## What's Included

### 1. Probe Registry (`probing/registry.py`)

Import-time dependency checking. The `AVAILABLE_PROBES` dict gates the entire system — the orchestrator checks it before scheduling any probe.

- **18 core probes** always available (PyTorch + sklearn + scipy)
- **3 optional probes** (TDA, SINDy, CEBRA) with one-time warning on first skip

```python
from l5pc.probing.registry import AVAILABLE_PROBES, is_available
print(is_available('tda'))  # False if ripser not installed
```

### 2. MLP Nonlinear Probing Control (`probing/mlp_probe.py`)

Every Ridge probe MUST have an MLP companion. Catches nonlinear encoding that Ridge misses.

- `MLPProbe` — 2-layer MLP with controlled capacity (hidden_dim=64 max, per Hewitt & Liang 2019)
- `mlp_delta_r2()` — Computes both Ridge and MLP delta-R2 with 5-fold CV
- `_classify_encoding()` — Classifies as LINEAR_ENCODED / NONLINEAR_ENCODED / NONLINEAR_ONLY / ZOMBIE / AMBIGUOUS

**Key insight:** If MLP delta-R2 >> Ridge delta-R2, the target is nonlinearly encoded, NOT a zombie. Ridge would falsely report it as zombie.

```python
from l5pc.probing.mlp_probe import mlp_delta_r2
results = mlp_delta_r2(h_trained, h_untrained, targets, names)
# results['gNaTa_t']['encoding_type'] -> 'NONLINEAR_ENCODED'
```

### 3. SAE Superposition Decomposition (`probing/sae_probe.py`)

TopK Sparse Autoencoder (Gao et al. 2024). Detects variables that are encoded but entangled (superposed) in the hidden state.

- `SparseAutoencoder` — TopK activation (k=20) for direct sparsity control
- `train_sae()` — Train on frozen hidden state trajectories
- `sae_probe_biological_variables()` — Two-stage decomposition + Ridge probing + monosemanticity scoring

**Key insight:** If raw Ridge gives low R2 but SAE->Ridge gives high R2, the variable is SUPERPOSED — encoded but invisible to linear probes without decomposition.

| SAE R2 | Raw Ridge R2 | Interpretation |
|--------|-------------|----------------|
| High | High | Monosemantic encoding |
| High | Low | **SUPERPOSITION DETECTED** |
| Low | Low | Genuinely not encoded (zombie) |

```python
from l5pc.probing.sae_probe import train_sae, sae_probe_biological_variables
sae, loss = train_sae(hidden_states, 128, expansion_factor=4)
results = sae_probe_biological_variables(sae, hidden_states, bio_targets, names)
print(results['superposition_detected'])  # {var_name: True/False}
```

### 4. Statistical Hardening Suite (`probing/hardening/`)

13-method suite providing formal significance testing for every delta-R2 value. Split into 5 sub-modules + orchestrator:

| File | Methods | What It Does |
|------|---------|-------------|
| `permutation.py` | 1-3 | Block permutation, IAAFT phase randomization, circular shift null |
| `diagnostics.py` | 4, 7, 13 | Effective DOF (Bartlett), Durbin-Watson, Ljung-Box |
| `corrections.py` | 5, 9, 10 | FDR (BH), TOST equivalence testing, Bayes factor |
| `frequency.py` | 6, 8 | Frequency-resolved R2, partial coherence conditioning |
| `gap_cv.py` | 11, 12 | Gap temporal CV, cluster permutation testing |
| `__init__.py` | All | `hardened_probe()` orchestrator + `_hardened_verdict()` |

```python
from l5pc.probing.hardening import hardened_probe
result = hardened_probe(h_trained, h_untrained, target, 'gNaTa_t')
print(result['hardened_verdict'])       # 'CONFIRMED_ENCODED'
print(result['p_block_permutation'])    # 0.002
print(result['tost_zombie'])            # {'zombie_confirmed': False, ...}
print(result['bayes_factor'])           # {'bf01': 0.03, 'interpretation': 'STRONG_NON_ZOMBIE'}
```

### 5. Configuration (`config.py` additions)

47 new lines of constants appended to the existing centralised config:

- SAE: expansion factors, k, learning rate, epochs, batch size, dead feature threshold
- MLP probe: hidden dim, epochs, learning rate, nonlinear gain threshold
- Hardening: permutation counts, block size, FDR alpha, TOST bounds, BF prior scale, gap size
- Frequency bands: ultra_slow (0.1-1 Hz), slow (1-10), medium (10-100), fast (100-450)
- Probe tiers: CCA components, RSA samples, Koopman modes, DSA params, TDA params, MINE/MDL params, temporal window sizes

## Files Created

| File | Lines | Description |
|------|-------|-------------|
| `l5pc/probing/registry.py` | 71 | Probe availability registry |
| `l5pc/probing/mlp_probe.py` | 167 | MLP nonlinear probing control |
| `l5pc/probing/sae_probe.py` | 218 | SAE superposition decomposition |
| `l5pc/probing/hardening/__init__.py` | ~120 | Hardened probe orchestrator |
| `l5pc/probing/hardening/permutation.py` | ~110 | Null distribution generators |
| `l5pc/probing/hardening/diagnostics.py` | ~60 | Residual diagnostics |
| `l5pc/probing/hardening/corrections.py` | ~90 | Multiple comparison corrections |
| `l5pc/probing/hardening/frequency.py` | ~70 | Frequency-domain validation |
| `l5pc/probing/hardening/gap_cv.py` | ~90 | Temporal gap CV + cluster testing |
| `l5pc/config.py` | +47 | v3.0 configuration constants |
| `tests/test_phase1_smoke.py` | 77 | 6 smoke tests |
| `requirements-v3.txt` | 15 | Optional dependencies |

## Dependencies

**Required (already installed):** torch, scikit-learn, scipy, numpy

**Optional (for later phases):** ripser, persim, pysindy, cebra, anthropic
