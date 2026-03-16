# Phase 3: Extended Probes — Topological, Information-Theoretic, Temporal

**Status: Planned (Tasks 11-13)**

Phase 3 adds the remaining probe tiers that provide complementary evidence for the zombie verdict. These are higher-tier probes that run only on surrogates that pass earlier tiers.

## What Will Be Included

### Task 11: Topological Probes (`probing/topological_probes.py`)

Persistent homology comparison using TDA — tests whether the topology of the hidden state manifold matches biological state space topology.

| Probe | Method | What It Tests |
|-------|--------|---------------|
| TDA | Persistent homology (ripser) | Shared topological features (holes, voids) |

Requires `ripser` and `persim`. Degrades gracefully via registry.

### Task 12: Information-Theoretic Probes (`probing/information_probes.py`)

Tests information content without assuming linear relationships.

| Probe | Method | What It Tests |
|-------|--------|---------------|
| MINE | Mutual Information Neural Estimation | Model-free MI between hidden and bio |
| MDL | Minimum Description Length | Compression-based encoding measure |

MINE uses a small neural network (PyTorch). MDL uses sklearn. Both always available.

### Task 13: Temporal and Structural Probes (`probing/temporal_probes.py`)

Tests temporal encoding structure — when and how information flows through the network.

| Probe | Method | What It Tests |
|-------|--------|---------------|
| Temporal windows | h(t-k:t) -> bio(t) | How far back does the network look? |
| Generalization matrices | Train@t, test@t' | Temporal stability of encoding |
| Gate-specific | Probe LSTM f/i/o/g gates | Which gate carries which information? |

All use PyTorch + sklearn. Always available.

## Dependencies

**Required:** PyTorch, sklearn, scipy, numpy (always available)

**Optional:** ripser, persim (TDA only)

## Tiered Evaluation Context

In the full factory (Phase 4), probes run in tiers with early termination:
- **Tier 0:** Ridge + MLP (Phase 1) — fast screen
- **Tier 1:** SAE + CCA + RSA (Phase 1-2) — structural probes
- **Tier 2:** Resample ablation (existing) — causal confirmation
- **Tier 3-6:** Dynamical, topological, info-theoretic, temporal (Phase 2-3)

Surrogates that fail Tier 0 never reach Tier 3+. This keeps compute bounded.
