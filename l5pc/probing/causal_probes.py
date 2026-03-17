"""
L5PC DESCARTES -- Tier 5: Causal Probes

DAS (Distributed Alignment Search) and Transfer Entropy.

NOTE: Resample ablation already exists in ablation.py — NOT duplicated here.
"""

import logging

import numpy as np
import torch
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 9.2  DAS (Distributed Alignment Search)
# ──────────────────────────────────────────────────────────────

def distributed_alignment_search(hidden, target, model,
                                   n_directions=100, device='cpu'):
    """
    DAS (Geiger et al.): Find the direction in hidden space
    that maximally encodes the target, then ablate along that direction.

    Unlike per-dimension ablation, DAS finds rotated encodings
    that span multiple hidden dimensions.
    """
    # Find encoding direction via Ridge
    ridge = Ridge(alpha=1.0)
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_target = target.ravel()
    ridge.fit(flat_hidden, flat_target)

    # Encoding direction = Ridge coefficients (normalized)
    direction = ridge.coef_ / (np.linalg.norm(ridge.coef_) + 1e-10)

    # Project hidden states onto encoding direction
    projections = flat_hidden @ direction

    # Ablate along this direction (resample the projection)
    rng = np.random.default_rng(42)

    ablated_ccs = []
    for _ in range(50):
        shuffled_proj = rng.permutation(projections)
        hidden_ablated = flat_hidden - np.outer(projections, direction) + \
                         np.outer(shuffled_proj, direction)

        # Reshape and re-run output layer
        hidden_reshaped = hidden_ablated.reshape(hidden.shape)
        h_tensor = torch.tensor(hidden_reshaped, dtype=torch.float32, device=device)
        with torch.no_grad():
            output = model.decode(h_tensor)
        ablated_ccs.append(_cross_condition_cc(output.cpu().numpy(), target))

    return {
        'encoding_direction': direction,
        'projection_r2': float(ridge.score(flat_hidden, flat_target)),
        'ablation_cc_mean': float(np.mean(ablated_ccs)),
        'ablation_cc_std': float(np.std(ablated_ccs)),
    }


# ──────────────────────────────────────────────────────────────
# 9.3  Transfer Entropy
# ──────────────────────────────────────────────────────────────

def transfer_entropy(source, target, lag=1, n_bins=10):
    """
    Directed information flow: does source causally predict target
    beyond target's own history?

    TE(X->Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})
    """
    from sklearn.metrics import mutual_info_score

    # Discretize
    source_binned = np.digitize(source, np.linspace(source.min(), source.max(), n_bins))
    target_binned = np.digitize(target, np.linspace(target.min(), target.max(), n_bins))

    # Lagged versions
    T = len(source) - lag
    Y_t = target_binned[lag:]
    Y_past = target_binned[:T]
    X_past = source_binned[:T]

    # Joint coding for MI estimation
    YX_joint = Y_past * (n_bins + 1) + X_past

    # TE = MI(Y_t; X_past | Y_past)
    # Approximate via: MI(Y_t; (Y_past, X_past)) - MI(Y_t; Y_past)
    mi_joint = mutual_info_score(Y_t, YX_joint)
    mi_past = mutual_info_score(Y_t, Y_past)

    te = max(0, mi_joint - mi_past)

    return {'transfer_entropy': float(te), 'lag': lag}


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────

def _cross_condition_cc(predicted, actual):
    """Cross-condition correlation: mean Pearson r across conditions."""
    if predicted.ndim == 3:
        predicted = predicted.mean(axis=1)
        actual = actual.mean(axis=1)
    return float(np.corrcoef(predicted.ravel(), actual.ravel())[0, 1])
