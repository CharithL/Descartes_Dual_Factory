"""
L5PC DESCARTES -- Tier 6: Information-Theoretic Probes

MINE (Mutual Information Neural Estimation) and MDL (Minimum Description Length).

MINE: captures nonlinear statistical dependencies that R^2 misses.
MDL: measures compression — how many bits to communicate target given hidden states.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 10.1  MINE (Mutual Information Neural Estimation)
# ──────────────────────────────────────────────────────────────

class MINEProbe(nn.Module):
    """
    MINE (Belghazi et al. 2018): neural estimation of mutual information.
    Captures nonlinear statistical dependencies that R^2 misses.

    Caveat: Poole et al. 2019 showed MINE saturates for high MI.
    Use InfoNCE bound as alternative for high-dimensional targets.
    """

    def __init__(self, x_dim, y_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1))


def mine_mutual_information(hidden, target, epochs=200,
                             batch_size=512, device='cpu'):
    """Estimate MI(hidden; target) via MINE."""
    hidden_t = torch.tensor(hidden, dtype=torch.float32, device=device)
    target_t = torch.tensor(target.reshape(-1, 1), dtype=torch.float32, device=device)

    model = MINEProbe(hidden.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    N = len(hidden)
    mi_estimates = []

    for epoch in range(epochs):
        idx = torch.randperm(N, device=device)[:batch_size]

        # Joint samples
        t_joint = model(hidden_t[idx], target_t[idx])

        # Marginal (shuffle target)
        idx_marginal = torch.randperm(N, device=device)[:batch_size]
        t_marginal = model(hidden_t[idx], target_t[idx_marginal])

        # DV bound
        mi = t_joint.mean() - torch.log(torch.exp(t_marginal).mean() + 1e-10)

        loss = -mi  # Maximize MI
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mi_estimates.append(float(mi))

    # Take last 50 epochs average
    final_mi = np.mean(mi_estimates[-50:])

    return {'mutual_information': float(final_mi),
            'mi_trajectory': mi_estimates}


# ──────────────────────────────────────────────────────────────
# 10.2  MDL Probing (Voita & Titov 2020)
# ──────────────────────────────────────────────────────────────

def mdl_probe(hidden, target, n_portions=10):
    """
    Minimum Description Length probing.

    Instead of accuracy, measures how many bits are needed to
    communicate the target given the hidden states.
    Controls for probe complexity automatically.

    Lower codelength = better encoding.
    """
    N = len(hidden)
    portion_sizes = [int(N * (i + 1) / n_portions) for i in range(n_portions)]

    codelengths = []
    for portion_end in portion_sizes:
        X_portion = hidden[:portion_end]
        y_portion = target[:portion_end]

        # Prequential code: train on first k, predict k+1
        kf = KFold(n_splits=5, shuffle=False)
        losses = []
        for train_idx, test_idx in kf.split(X_portion):
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_portion[train_idx], y_portion[train_idx])
            pred = ridge.predict(X_portion[test_idx])
            mse = np.mean((pred - y_portion[test_idx])**2)
            # Codelength ~ -log likelihood ~ MSE under Gaussian
            losses.append(0.5 * np.log(mse + 1e-10) * len(test_idx))

        codelengths.append(np.sum(losses))

    # Online codelength = sum of incremental costs
    total_codelength = codelengths[-1]

    # Uniform baseline (no encoding)
    var_target = np.var(target) + 1e-10
    uniform_codelength = 0.5 * np.log(var_target) * N

    compression = 1 - (total_codelength / uniform_codelength)

    return {
        'total_codelength': float(total_codelength),
        'uniform_codelength': float(uniform_codelength),
        'compression_ratio': float(compression),
        'encoded': compression > 0.1
    }
