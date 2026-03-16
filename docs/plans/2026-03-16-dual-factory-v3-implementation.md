# Dual Factory v3.0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the complete DESCARTES Dual Factory v3.0 — 43 probe methods, 13-method statistical hardening, SAE superposition analysis, surrogate genome evolution with LLM balloon expansion, and unified zombie verdict generator.

**Architecture:** Registry-driven modular probing (C1) co-evolving with genome-based surrogate search (C2). The probing factory is the inner loop (fitness evaluator), the surrogate factory is the outer loop (architecture search). Thompson sampling allocates compute, DreamCoder synthesizes design patterns, LLM balloon expands when search stalls.

**Tech Stack:** Python 3.10+, PyTorch, scikit-learn, scipy, numpy. Optional: ripser, persim, pysindy, cebra, anthropic SDK.

**Source of truth:** `DESCARTES_DUAL_FACTORY_V3 LLM(1).md` — all code is transcribed verbatim from that guide.

**Base path:** `C:/Users/chari/OneDrive/Documents/Descartes_Cogito/L5PC/`

---

## PHASE 1: HIGH PRIORITY (Tasks 1-7)

### Task 1: Probe Registry

**Files:**
- Create: `l5pc/probing/registry.py`

**Step 1: Create the registry module**

```python
"""
L5PC DESCARTES -- Probe Availability Registry

Import-time dependency checking for optional packages.
One WARNING per session per missing package, then silent.
AVAILABLE_PROBES dict gates the orchestrator.
"""

import logging

logger = logging.getLogger(__name__)

AVAILABLE_PROBES = {}
_WARNED = set()


def _check_optional(probe_name, package_name, install_hint):
    """Check if an optional dependency is available. Warn once if not."""
    global AVAILABLE_PROBES
    try:
        __import__(package_name)
        AVAILABLE_PROBES[probe_name] = True
    except ImportError:
        AVAILABLE_PROBES[probe_name] = False
        if probe_name not in _WARNED:
            logger.warning(
                "WARNING: %s not installed — %s probes disabled (%s)",
                package_name, probe_name, install_hint,
            )
            _WARNED.add(probe_name)


def is_available(probe_name):
    """Check if a probe is available for scheduling."""
    return AVAILABLE_PROBES.get(probe_name, False)


# === Core probes (always available — PyTorch + sklearn + scipy) ===
AVAILABLE_PROBES['ridge'] = True
AVAILABLE_PROBES['mlp'] = True
AVAILABLE_PROBES['sae'] = True
AVAILABLE_PROBES['hardening'] = True
AVAILABLE_PROBES['resample_ablation'] = True
AVAILABLE_PROBES['cca'] = True      # sklearn.cross_decomposition
AVAILABLE_PROBES['rsa'] = True      # scipy.spatial.distance
AVAILABLE_PROBES['cka'] = True      # numpy only
AVAILABLE_PROBES['koopman'] = True  # scipy.linalg + numpy
AVAILABLE_PROBES['dsa'] = True      # numpy.linalg
AVAILABLE_PROBES['mine'] = True     # PyTorch
AVAILABLE_PROBES['mdl'] = True      # sklearn + numpy
AVAILABLE_PROBES['temporal'] = True  # sklearn + numpy
AVAILABLE_PROBES['gate_specific'] = True  # PyTorch
AVAILABLE_PROBES['transfer_entropy'] = True  # sklearn.metrics
AVAILABLE_PROBES['das'] = True      # sklearn + PyTorch
AVAILABLE_PROBES['frequency'] = True  # scipy.signal
AVAILABLE_PROBES['pi_vae'] = True   # PyTorch

# === Optional probes (require extra packages) ===
_check_optional('tda', 'ripser', 'pip install ripser persim')
_check_optional('sindy', 'pysindy', 'pip install pysindy')
_check_optional('cebra', 'cebra', 'pip install cebra')


def get_available_probe_names():
    """Return list of all available probe names."""
    return [name for name, available in AVAILABLE_PROBES.items() if available]


def get_unavailable_probe_names():
    """Return list of all unavailable probe names."""
    return [name for name, available in AVAILABLE_PROBES.items() if not available]
```

**Step 2: Verify import works**

Run: `cd L5PC && python -c "from l5pc.probing.registry import AVAILABLE_PROBES; print(AVAILABLE_PROBES)"`
Expected: Dict with core probes True, optional probes True/False depending on install

**Step 3: Commit**

```
git add l5pc/probing/registry.py
git commit -m "feat: add probe availability registry with one-time warnings"
```

---

### Task 2: MLP Probe

**Files:**
- Create: `l5pc/probing/mlp_probe.py`

**Step 1: Create MLP probe module**

Transcribe from guide Section 4.2. Key classes/functions:
- `MLPProbe(nn.Module)` — 2-layer MLP with controlled capacity (hidden_dim=64 max)
- `mlp_delta_r2()` — Compute MLP delta-R2 alongside Ridge delta-R2 for all targets
- `_classify_encoding()` — LINEAR_ENCODED / NONLINEAR_ONLY / NONLINEAR_ENCODED / ZOMBIE / AMBIGUOUS
- `_train_mlp_fold()` — Train MLP on one CV fold

```python
"""
L5PC DESCARTES -- MLP Delta-R2 Nonlinear Probing Control

Rule: every Ridge probe MUST have an MLP companion.
If MLP delta-R2 >> Ridge delta-R2: target is nonlinearly encoded, not zombie.
If MLP delta-R2 approx Ridge delta-R2: linear probing is sufficient.

Capacity control (Hewitt and Liang 2019):
  - Hidden dim = 64 maximum (not 256 or 512)
  - 2 layers maximum
  - delta-R2 (trained minus untrained) is the metric, not raw R2
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from l5pc.config import MLP_PROBE_HIDDEN_DIM, MLP_PROBE_EPOCHS, MLP_PROBE_LR

logger = logging.getLogger(__name__)


class MLPProbe(nn.Module):
    """2-layer MLP probe with controlled capacity."""

    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = MLP_PROBE_HIDDEN_DIM
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_delta_r2(hidden_trained, hidden_untrained, targets,
                 target_names, hidden_dim=None, epochs=None,
                 lr=None, n_splits=5, device='cpu'):
    """
    Compute MLP delta-R2 alongside Ridge delta-R2 for all targets.

    Returns comparison table showing which targets are
    nonlinearly encoded (MLP >> Ridge) vs truly zombie (both low).
    """
    if hidden_dim is None:
        hidden_dim = MLP_PROBE_HIDDEN_DIM
    if epochs is None:
        epochs = MLP_PROBE_EPOCHS
    if lr is None:
        lr = MLP_PROBE_LR

    results = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for j, name in enumerate(target_names):
        target = targets[:, j] if targets.ndim > 1 else targets

        ridge_trained_scores = []
        ridge_untrained_scores = []
        mlp_trained_scores = []
        mlp_untrained_scores = []

        for train_idx, test_idx in kf.split(hidden_trained):
            # Ridge trained
            ridge = Ridge(alpha=1.0)
            ridge.fit(hidden_trained[train_idx], target[train_idx])
            ridge_trained_scores.append(
                ridge.score(hidden_trained[test_idx], target[test_idx]))

            # Ridge untrained
            ridge_u = Ridge(alpha=1.0)
            ridge_u.fit(hidden_untrained[train_idx], target[train_idx])
            ridge_untrained_scores.append(
                ridge_u.score(hidden_untrained[test_idx], target[test_idx]))

            # MLP trained
            mlp_t = _train_mlp_fold(
                hidden_trained[train_idx], target[train_idx],
                hidden_trained[test_idx], target[test_idx],
                hidden_trained.shape[1], hidden_dim, epochs, lr, device)
            mlp_trained_scores.append(mlp_t)

            # MLP untrained
            mlp_u = _train_mlp_fold(
                hidden_untrained[train_idx], target[train_idx],
                hidden_untrained[test_idx], target[test_idx],
                hidden_untrained.shape[1], hidden_dim, epochs, lr, device)
            mlp_untrained_scores.append(mlp_u)

        ridge_delta = np.mean(ridge_trained_scores) - np.mean(ridge_untrained_scores)
        mlp_delta = np.mean(mlp_trained_scores) - np.mean(mlp_untrained_scores)

        results[name] = {
            'ridge_trained': float(np.mean(ridge_trained_scores)),
            'ridge_untrained': float(np.mean(ridge_untrained_scores)),
            'ridge_delta': float(ridge_delta),
            'mlp_trained': float(np.mean(mlp_trained_scores)),
            'mlp_untrained': float(np.mean(mlp_untrained_scores)),
            'mlp_delta': float(mlp_delta),
            'nonlinear_gain': float(mlp_delta - ridge_delta),
            'encoding_type': _classify_encoding(ridge_delta, mlp_delta),
        }

        logger.info(
            "  %s: ridge_dR2=%.3f  mlp_dR2=%.3f  gain=%.3f  [%s]",
            name, ridge_delta, mlp_delta, mlp_delta - ridge_delta,
            results[name]['encoding_type'],
        )

    return results


def _classify_encoding(ridge_delta, mlp_delta, threshold=0.05):
    """Classify encoding type from Ridge vs MLP comparison."""
    if ridge_delta > threshold and mlp_delta > threshold:
        if mlp_delta > ridge_delta + 0.1:
            return 'NONLINEAR_ENCODED'
        return 'LINEAR_ENCODED'
    elif mlp_delta > threshold and ridge_delta <= threshold:
        return 'NONLINEAR_ONLY'
    elif ridge_delta <= threshold and mlp_delta <= threshold:
        return 'ZOMBIE'
    else:
        return 'AMBIGUOUS'


def _train_mlp_fold(X_train, y_train, X_test, y_test,
                    input_dim, hidden_dim, epochs, lr, device):
    """Train MLP probe on one fold and return test R2."""
    model = MLPProbe(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_te = torch.tensor(y_test, dtype=torch.float32, device=device)

    # Normalize targets
    y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-8
    y_tr_norm = (y_tr - y_mean) / y_std

    model.train()
    for epoch in range(epochs):
        pred = model(X_tr)
        loss = nn.functional.mse_loss(pred, y_tr_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_test = model(X_te) * y_std + y_mean
        ss_res = ((pred_test - y_te) ** 2).sum()
        ss_tot = ((y_te - y_te.mean()) ** 2).sum()
        r2 = 1.0 - (ss_res / (ss_tot + 1e-10))

    return float(r2.cpu())
```

**Step 2: Verify import**

Run: `cd L5PC && python -c "from l5pc.probing.mlp_probe import MLPProbe, mlp_delta_r2; print('OK')"`

**Step 3: Commit**

```
git add l5pc/probing/mlp_probe.py
git commit -m "feat: add MLP delta-R2 nonlinear probing control"
```

---

### Task 3: SAE Probe

**Files:**
- Create: `l5pc/probing/sae_probe.py`

Transcribe from guide Section 3.2. Key classes/functions:
- `SparseAutoencoder(nn.Module)` — TopK SAE (Gao et al. 2024)
- `train_sae()` — Train on frozen hidden states
- `sae_probe_biological_variables()` — Two-stage SAE decomposition + Ridge probing

The code is exactly as shown in guide lines 214-404. Import SAE hyperparameters from config.

**Step 1: Create the file** with full code from guide Section 3.2.

**Step 2: Verify import**

Run: `cd L5PC && python -c "from l5pc.probing.sae_probe import SparseAutoencoder, train_sae; print('OK')"`

**Step 3: Commit**

```
git add l5pc/probing/sae_probe.py
git commit -m "feat: add SAE superposition decomposition probing"
```

---

### Task 4: Statistical Hardening Sub-Package

**Files:**
- Create: `l5pc/probing/hardening/__init__.py`
- Create: `l5pc/probing/hardening/permutation.py`
- Create: `l5pc/probing/hardening/diagnostics.py`
- Create: `l5pc/probing/hardening/corrections.py`
- Create: `l5pc/probing/hardening/frequency.py`
- Create: `l5pc/probing/hardening/gap_cv.py`

#### Step 1: Create `permutation.py`

From guide Section 5.2, Methods 1-3:
- `block_permute(y, block_size, rng)`
- `adaptive_block_size(target, sampling_rate_hz=1000)`
- `phase_randomize(signal, rng)` (IAAFT)
- `circular_shift_null(hidden, target, n_shifts, min_shift_factor, rng)`

#### Step 2: Create `diagnostics.py`

From guide Methods 4, 7, 13:
- `effective_dof(x, y)` — Bartlett's formula
- `durbin_watson(residuals)`
- `ljung_box_residual_test(residuals, n_lags=20)`

#### Step 3: Create `corrections.py`

From guide Methods 5, 9, 10:
- `fdr_correction(p_values, alpha=0.05)`
- `tost_zombie_test(delta_r2, se_delta_r2, n_eff, equivalence_bound)`
- `bayes_factor_null(delta_r2, se_delta_r2, prior_scale)`

#### Step 4: Create `frequency.py`

From guide Methods 6, 8:
- `frequency_resolved_r2(hidden, target, fs, bands)`
- `partial_coherence_r2(hidden, target, input_signal)`

#### Step 5: Create `gap_cv.py`

From guide Methods 11, 12:
- `gap_temporal_cv(hidden, target, n_splits, gap_size)`
- `cluster_permutation_test(delta_r2_map, n_permutations, cluster_threshold, rng)`

#### Step 6: Create `__init__.py`

From guide Section 5.3:
- `hardened_probe()` — runs ONE target through the complete 13-method suite
- `_hardened_verdict()` — generate definitive verdict from all statistical evidence

Imports from all sub-modules and from `l5pc.probing.mlp_probe`.

#### Step 7: Verify

Run: `cd L5PC && python -c "from l5pc.probing.hardening import hardened_probe; print('OK')"`

#### Step 8: Commit

```
git add l5pc/probing/hardening/
git commit -m "feat: add 13-method statistical hardening suite"
```

---

### Task 5: Config Additions

**Files:**
- Modify: `l5pc/config.py` — append v3.0 constants at end of file

Add all constants from the design doc's "Config Additions" section:
SAE params, MLP probe params, hardening params, frequency bands, probe tier params.

**Step 1: Append constants to config.py**

**Step 2: Verify**

Run: `cd L5PC && python -c "from l5pc.config import SAE_K, MLP_PROBE_HIDDEN_DIM, HARDENING_N_BLOCK_PERMS; print('OK')"`

**Step 3: Commit**

```
git add l5pc/config.py
git commit -m "feat: add v3.0 configuration constants (SAE, MLP, hardening, probe tiers)"
```

---

### Task 6: Phase 1 Integration Smoke Test

**Step 1: Write a quick integration test**

Create `tests/test_phase1_smoke.py`:

```python
"""Smoke test: Phase 1 modules import and run on synthetic data."""
import numpy as np
import torch


def test_registry_core_probes_available():
    from l5pc.probing.registry import AVAILABLE_PROBES
    for core in ['ridge', 'mlp', 'sae', 'hardening', 'resample_ablation']:
        assert AVAILABLE_PROBES[core] is True, f"{core} should be available"


def test_mlp_probe_synthetic():
    from l5pc.probing.mlp_probe import mlp_delta_r2
    rng = np.random.RandomState(42)
    N, D = 100, 16
    hidden_trained = rng.randn(N, D).astype(np.float32)
    hidden_untrained = rng.randn(N, D).astype(np.float32)
    # Target correlates with first dim of trained
    target = hidden_trained[:, 0] + 0.1 * rng.randn(N)
    targets = target.reshape(-1, 1).astype(np.float32)
    results = mlp_delta_r2(hidden_trained, hidden_untrained, targets,
                           ['test_var'], epochs=5, n_splits=3)
    assert 'test_var' in results
    assert 'encoding_type' in results['test_var']


def test_sae_synthetic():
    from l5pc.probing.sae_probe import SparseAutoencoder, train_sae
    rng = np.random.RandomState(42)
    hidden = [rng.randn(50, 16).astype(np.float32)]
    sae, loss_history = train_sae(hidden, 16, expansion_factor=2,
                                   k=5, epochs=3, batch_size=32)
    assert len(loss_history) == 3
    assert loss_history[-1] < loss_history[0]  # Loss decreases


def test_hardening_imports():
    from l5pc.probing.hardening import hardened_probe
    from l5pc.probing.hardening.permutation import block_permute, adaptive_block_size
    from l5pc.probing.hardening.diagnostics import durbin_watson, effective_dof
    from l5pc.probing.hardening.corrections import fdr_correction, tost_zombie_test
    from l5pc.probing.hardening.frequency import frequency_resolved_r2
    from l5pc.probing.hardening.gap_cv import gap_temporal_cv
    # All imports succeed
    assert callable(hardened_probe)
```

**Step 2: Run tests**

Run: `cd L5PC && python -m pytest tests/test_phase1_smoke.py -v`

**Step 3: Commit**

```
git add tests/test_phase1_smoke.py
git commit -m "test: Phase 1 smoke tests for registry, MLP, SAE, hardening"
```

---

### Task 7: Requirements File

**Files:**
- Create: `requirements-v3.txt`

```
# DESCARTES Dual Factory v3.0 — Optional Dependencies
# Core deps (torch, scikit-learn, scipy, numpy) assumed installed

# Tier 3: Dynamical probes
pysindy>=2.0

# Tier 4: Topological probes
ripser>=0.6
persim>=0.3

# Tier 2: Joint alignment (optional)
cebra>=0.4

# LLM balloon expansion
anthropic>=0.40
```

**Commit:**

```
git add requirements-v3.txt
git commit -m "feat: add requirements-v3.txt for optional dependencies"
```

---

## PHASE 2: CORE PROBES (Tasks 8-10)

### Task 8: Joint Alignment Probes

**Files:**
- Create: `l5pc/probing/joint_alignment.py`

Transcribe from guide Sections 6.1-6.5:
- `cca_alignment()` — Cross-validated CCA with block-permutation null
- `rsa_comparison()` — RSA with Spearman correlation between RDMs
- `cka_comparison()` — CKA with RBF and linear kernels
- `PiVAE(nn.Module)` — Identifiable conditional latent recovery
- `cebra_alignment()` — CEBRA joint embedding (graceful fallback)
- `procrustes_alignment()` — Procrustes rotation + scaling

CEBRA wraps in try/except ImportError. Uses `from l5pc.probing.registry import is_available` before attempting.

**Verify:** `python -c "from l5pc.probing.joint_alignment import cca_alignment, rsa_comparison, cka_comparison; print('OK')"`

**Commit:** `git commit -m "feat: add joint alignment probes (CCA, RSA, CKA, pi-VAE, CEBRA)"`

---

### Task 9: Causal Probes

**Files:**
- Create: `l5pc/probing/causal_probes.py`

Transcribe from guide Sections 9.2-9.3:
- `distributed_alignment_search()` — DAS (Geiger et al.)
- `transfer_entropy()` — Directed information flow
- `_cross_condition_cc()` — Helper

Note: Resample ablation already exists in `ablation.py` — do NOT duplicate.

**Verify:** `python -c "from l5pc.probing.causal_probes import distributed_alignment_search, transfer_entropy; print('OK')"`

**Commit:** `git commit -m "feat: add causal probes (DAS, transfer entropy)"`

---

### Task 10: Dynamical Probes

**Files:**
- Create: `l5pc/probing/dynamical_probes.py`

Transcribe from guide Sections 7.1-7.3:
- `koopman_spectral_comparison()` — DMD-based Koopman estimation
- `sindy_probe()` — SINDy symbolic regression (graceful fallback)
- `_compare_to_hh()` — Compare discovered equations to HH structure
- `dsa_comparison()` — DSA delay-embedded comparison

SINDy wraps in try/except ImportError.

**Verify:** `python -c "from l5pc.probing.dynamical_probes import koopman_spectral_comparison, dsa_comparison; print('OK')"`

**Commit:** `git commit -m "feat: add dynamical probes (Koopman, SINDy, DSA)"`

---

## PHASE 3: EXTENDED PROBES (Tasks 11-13)

### Task 11: Topological Probes

**Files:**
- Create: `l5pc/probing/topological_probes.py`

From guide Section 8.1:
- `tda_comparison()` — Persistent homology with ripser (graceful fallback)

**Commit:** `git commit -m "feat: add topological probes (TDA/persistent homology)"`

---

### Task 12: Information-Theoretic Probes

**Files:**
- Create: `l5pc/probing/information_probes.py`

From guide Sections 10.1-10.2:
- `MINEProbe(nn.Module)` — MINE neural MI estimation
- `mine_mutual_information()` — Estimate MI via MINE
- `mdl_probe()` — Minimum description length probing

**Commit:** `git commit -m "feat: add information-theoretic probes (MINE, MDL)"`

---

### Task 13: Temporal and Structural Probes

**Files:**
- Create: `l5pc/probing/temporal_probes.py`

From guide Sections 11.1-11.3:
- `temporal_probe()` — h(t-k:t) to bio(t) with multiple window sizes
- `temporal_generalization()` — Train at t, test at t-prime matrix
- `gate_specific_probe()` — Probe LSTM forget/input/output gates separately

**Commit:** `git commit -m "feat: add temporal and structural probes (windows, gen matrices, gate-specific)"`

---

## PHASE 4: INTEGRATION (Tasks 14-24)

### Task 14: Factory Config

**Files:**
- Create: `l5pc/factory/__init__.py`
- Create: `l5pc/factory/config.py`

Factory-specific constants: Thompson sampling priors, DreamCoder schedule, LLM balloon threshold, output CC gate, fitness weights, campaign phase boundaries, architecture options, hidden dim options, bio loss weight options.

**Commit:** `git commit -m "feat: add factory config with Thompson/DreamCoder/balloon constants"`

---

### Task 15: Probe Genome

**Files:**
- Create: `l5pc/factory/probe_genome.py`

From guide Section 2.2:
- `ProbeGenome_v3` dataclass with all fields

**Commit:** `git commit -m "feat: add ProbeGenome_v3 dataclass"`

---

### Task 16: Surrogate Genome

**Files:**
- Create: `l5pc/factory/surrogate_genome.py`

From guide Sections 13.3-13.4:
- `SurrogateGenome_v3` dataclass (full specification)
- `SurrogateGenomeComposer` — `compose_random()`, `mutate()`, `crossover()`

**Commit:** `git commit -m "feat: add SurrogateGenome_v3 and genome composer"`

---

### Task 17: Surrogate Trainer

**Files:**
- Create: `l5pc/factory/surrogate_trainer.py`

From guide Section 13.5:
- `SurrogateTrainer` — `train_and_validate()`, `_build_model()`, `_train()`, `_compute_regularization()`

**Commit:** `git commit -m "feat: add surrogate trainer with output validation gate"`

---

### Task 18: Surrogate Fitness

**Files:**
- Create: `l5pc/factory/surrogate_fitness.py`

From guide Section 13.6:
- `SurrogateFitness` — multi-objective fitness (alpha=output, beta=bio, gamma=causal)

**Commit:** `git commit -m "feat: add multi-objective surrogate fitness function"`

---

### Task 19: Zombie Verdict Generator

**Files:**
- Create: `l5pc/factory/verdict.py`

From guide Section 15.1:
- `ZombieVerdictGenerator_v3` — 8 verdict types, evidence bundle evaluation

**Commit:** `git commit -m "feat: add ZombieVerdictGenerator_v3 with 8 verdict types"`

---

### Task 20: LLM Balloon Expander

**Files:**
- Create: `l5pc/factory/llm_balloon.py`

From guide Sections 13.7-13.8:
- `SYSTEM_SURROGATE_BALLOON` prompt
- `SYSTEM_SURROGATE_GAP` prompt
- `SYSTEM_PROBE_BALLOON` prompt
- `SYSTEM_PROBE_GAP` prompt
- `LLMBalloonExpander` — `propose_novel_surrogates()`, `analyze_gaps()`, `_proposal_to_genome()`, `_call_llm()`

Uses `anthropic.Anthropic()` SDK instead of raw requests. Graceful fallback if anthropic not installed.

**Commit:** `git commit -m "feat: add LLM balloon expansion for surrogates and probes"`

---

### Task 21: DreamCoder

**Files:**
- Create: `l5pc/factory/dreamcoder.py`

From guide Section 13.9:
- `SurrogateDreamCoder` — `wake_phase()`, `sleep_phase()`, `compose_from_library()`, `_extract_patterns()`, `_apply_pattern()`

**Commit:** `git commit -m "feat: add DreamCoder wake-sleep pattern synthesis"`

---

### Task 22: Probing Factory Evaluator

**Files:**
- Create: `l5pc/factory/probing_evaluator.py`

From guide Section 14.2:
- `ProbingFactoryEvaluator` — tiered evaluation (Tier 0: Ridge+MLP, Tier 1: SAE+CCA+RSA, Tier 2: resample ablation), early termination

**Commit:** `git commit -m "feat: add ProbingFactoryEvaluator (inner loop with tiered evaluation)"`

---

### Task 23: Surrogate Factory

**Files:**
- Create: `l5pc/factory/surrogate_factory.py`

From guide Section 14.1:
- `SurrogateFactory` — `run_campaign()`, `_select_genome()` (4-phase), `_update_thompson()`, `_record_result()`, `_get_factory_state()`, `_detect_gaps()`, `_report_progress()`, `_final_report()`

**Commit:** `git commit -m "feat: add SurrogateFactory (outer loop with 4-phase campaign)"`

---

### Task 24: Dual Factory Orchestrator

**Files:**
- Create: `l5pc/factory/orchestrator.py`
- Create: `l5pc/surrogates/surrogate_registry.py`

From guide Section 14.3:
- `DualFactoryOrchestrator` — `run_full_campaign()`, `_cluster_architectures()` (DP clustering), `_generate_design_patterns()`

`surrogate_registry.py` dispatches `_build_model()` from genome:
- Maps `genome.architecture` to model class constructor
- Handles all 13 architecture types from guide Section 13.2

**Commit:** `git commit -m "feat: add DualFactoryOrchestrator and surrogate registry"`

---

## PHASE 5: FINAL INTEGRATION (Task 25)

### Task 25: Package Integration and Full Smoke Test

**Step 1: Update __init__.py files**

- `l5pc/probing/__init__.py` — re-export key functions
- `l5pc/surrogates/__init__.py` — re-export surrogate registry
- `l5pc/factory/__init__.py` — re-export orchestrator, verdict, evaluator

**Step 2: Create full integration smoke test**

```python
"""Full v3.0 integration test on synthetic data."""

def test_full_probing_pipeline_synthetic():
    """Test that all probe tiers can run on synthetic data."""
    from l5pc.probing.registry import get_available_probe_names
    available = get_available_probe_names()
    assert len(available) >= 15  # Core probes always available


def test_verdict_generator():
    from l5pc.factory.verdict import ZombieVerdictGenerator_v3
    gen = ZombieVerdictGenerator_v3()
    # Confirmed zombie case
    verdict = gen.generate_verdict({
        'ridge_delta_r2': 0.01, 'mlp_delta_r2': 0.02,
        'p_block_permutation': 0.8,
        'tost_zombie': {'zombie_confirmed': True},
        'bayes_factor': {'bf01': 15},
        'frequency_r2': {},
    })
    assert verdict['verdict'] == 'CONFIRMED_ZOMBIE'

    # Mandatory case
    verdict = gen.generate_verdict({
        'resample_ablation': {'causal': True},
    })
    assert verdict['verdict'] == 'MANDATORY'


def test_surrogate_genome_composition():
    from l5pc.factory.surrogate_genome import SurrogateGenome_v3, SurrogateGenomeComposer
    import numpy as np
    composer = SurrogateGenomeComposer()
    rng = np.random.default_rng(42)
    g1 = composer.compose_random(rng)
    assert g1.architecture in composer.ARCHITECTURES
    g2 = composer.mutate(g1, rng=rng)
    assert g2.generation == g1.generation + 1
    g3 = composer.crossover(g1, g2, rng=rng)
    assert len(g3.parent_ids) == 2


def test_factory_config_independent():
    """Factory config is separate from probing config."""
    from l5pc.factory.config import (
        FITNESS_ALPHA, FITNESS_BETA, FITNESS_GAMMA,
        LLM_BALLOON_THRESHOLD, OUTPUT_CC_THRESHOLD,
    )
    from l5pc.config import RIDGE_ALPHAS, CV_FOLDS
    # Both import successfully — separate namespaces
    assert abs(FITNESS_ALPHA + FITNESS_BETA + FITNESS_GAMMA - 1.0) < 1e-10
    assert isinstance(RIDGE_ALPHAS, list)
```

**Step 3: Run all tests**

Run: `cd L5PC && python -m pytest tests/ -v`

**Step 4: Final commit**

```
git add l5pc/ tests/
git commit -m "feat: complete DESCARTES Dual Factory v3.0 integration"
```

---

## Summary

| Phase | Tasks | New Files | Description |
|-------|-------|-----------|-------------|
| 1 | 1-7 | 9 | Registry, MLP, SAE, hardening (6 files), config, requirements |
| 2 | 8-10 | 3 | Joint alignment, causal probes, dynamical probes |
| 3 | 11-13 | 3 | Topological, information-theoretic, temporal probes |
| 4 | 14-24 | 12 | Factory package (config, genomes, trainer, fitness, verdict, LLM, DreamCoder, evaluator, factory, orchestrator, surrogate registry) |
| 5 | 25 | 0 | Integration, init updates, smoke tests |

**Total: 25 tasks, ~27 new files, ~4000 lines of code transcribed from guide.**
