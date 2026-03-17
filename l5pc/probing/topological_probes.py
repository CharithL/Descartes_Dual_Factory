"""
L5PC DESCARTES -- Tier 4: Topological Probes

Persistent homology / TDA — detects limit cycles, toroidal attractors,
and other topological features invisible to any linear/nonlinear probe.

Requires: pip install ripser persim
"""

import logging

import numpy as np

from l5pc.probing.registry import is_available

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 8.1  Persistent Homology / TDA
# ──────────────────────────────────────────────────────────────

def tda_comparison(hidden_trajectories, bio_trajectories,
                    max_dim=2, n_landmarks=200):
    """
    Compare topological structure (Betti numbers, persistence diagrams)
    between hidden state and biological manifolds.

    Detects: limit cycles (1-cycles), toroidal attractors (2-cycles),
    and other topological features invisible to any linear/nonlinear probe.

    Requires: pip install ripser persim
    """
    if not is_available('tda'):
        return {'error': 'ripser/persim not installed'}

    try:
        from ripser import ripser
        from persim import bottleneck

        # Subsample for tractability
        rng = np.random.default_rng(42)

        h_flat = np.concatenate(hidden_trajectories, axis=0)
        b_flat = np.concatenate(bio_trajectories, axis=0)

        if len(h_flat) > n_landmarks:
            idx_h = rng.choice(len(h_flat), n_landmarks, replace=False)
            idx_b = rng.choice(len(b_flat), n_landmarks, replace=False)
            h_flat = h_flat[idx_h]
            b_flat = b_flat[idx_b]

        # Compute persistence diagrams
        dgm_h = ripser(h_flat, maxdim=max_dim)['dgms']
        dgm_b = ripser(b_flat, maxdim=max_dim)['dgms']

        # Bottleneck distance per homology dimension
        distances = {}
        for d in range(max_dim + 1):
            dist = bottleneck(dgm_h[d], dgm_b[d])
            distances[f'H{d}'] = float(dist)

        return {
            'bottleneck_distances': distances,
            'mean_bottleneck': float(np.mean(list(distances.values()))),
            'topological_match': all(d < 1.0 for d in distances.values())
        }
    except ImportError:
        return {'error': 'ripser/persim not installed'}
