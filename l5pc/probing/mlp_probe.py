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
