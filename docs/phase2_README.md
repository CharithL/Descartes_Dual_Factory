# Phase 2: Core Probes — Joint Alignment, Causal, Dynamical

**Status: Planned (Tasks 8-10)**

Phase 2 adds Tier 2-3 probes that go beyond single-variable decoding to test structural, causal, and dynamical correspondence between surrogate and biological systems.

## What Will Be Included

### Task 8: Joint Alignment Probes (`probing/joint_alignment.py`)

Tests whether hidden states and biological variables share common structure — catches cases where no single variable is decodable but the joint geometry matches.

| Probe | Method | What It Tests |
|-------|--------|---------------|
| CCA | Canonical Correlation Analysis | Shared low-dimensional subspace |
| RSA | Representational Similarity Analysis | Geometry of distance matrices |
| CKA | Centered Kernel Alignment | Kernel-space similarity (linear + RBF) |
| pi-VAE | Identifiable conditional latent recovery | Disentangled latent structure |
| CEBRA | Joint embedding | Temporal alignment (optional dep) |
| Procrustes | Orthogonal alignment | Rotational correspondence |

CCA, RSA, CKA use block-permutation nulls from the hardening suite. CEBRA degrades gracefully if not installed.

### Task 9: Causal Probes (`probing/causal_probes.py`)

Tests directed information flow — correlation is not causation.

| Probe | Method | What It Tests |
|-------|--------|---------------|
| DAS | Distributed Alignment Search (Geiger et al.) | Causal correspondence via learned rotation |
| Transfer entropy | Directed information flow | h(t) -> bio(t+1) vs bio(t) -> h(t+1) |

Note: Resample ablation (the CANONICAL causal test) already exists in `ablation.py` and is NOT duplicated.

### Task 10: Dynamical Probes (`probing/dynamical_probes.py`)

Tests whether the surrogate's dynamics match biological dynamics — same equations of motion, not just same steady states.

| Probe | Method | What It Tests |
|-------|--------|---------------|
| Koopman | DMD-based spectral comparison | Shared eigenvalue structure |
| SINDy | Symbolic regression | Discovered equations match HH structure (optional dep) |
| DSA | Delay-embedded comparison | Attractor geometry |

SINDy degrades gracefully if pysindy not installed.

## Dependencies

All probes in this phase use PyTorch + sklearn + scipy (always available), except:
- CEBRA alignment requires `pip install cebra`
- SINDy probing requires `pip install pysindy`

Both degrade gracefully via the registry pattern from Phase 1.

## Integration Points

- All probes use `block_permute` from `hardening/permutation.py` for null distributions
- Registry (`AVAILABLE_PROBES`) gates CEBRA and SINDy probes
- Results feed into Phase 4's `ProbingFactoryEvaluator` tiered evaluation
