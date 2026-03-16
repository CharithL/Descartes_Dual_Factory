"""
L5PC DESCARTES -- SAE Superposition Decomposition Probing

TopK Sparse Autoencoder (Gao et al. 2024, OpenAI).
If raw Ridge gives low R2 but SAE Ridge gives high R2,
the variable is SUPERPOSED (encoded but entangled).

Architecture:
    encoder: h -> ReLU(topk(W_enc @ (h - b_dec) + b_enc))
    decoder: f -> W_dec @ f + b_dec

Loss:
    L = ||h - decoder(encoder(h))||^2 + 0.01 * ||decoder_norms - 1||^2
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from l5pc.config import SAE_K, SAE_LR, SAE_EPOCHS, SAE_BATCH_SIZE

logger = logging.getLogger(__name__)


class SparseAutoencoder(nn.Module):
    """
    TopK Sparse Autoencoder following Gao et al. (2024, OpenAI).

    Maps hidden_dim -> expansion_factor * hidden_dim sparse features.
    TopK activation gives direct sparsity control without tuning lambda.
    """

    def __init__(self, input_dim, expansion_factor=4, k=None):
        super().__init__()
        if k is None:
            k = SAE_K
        n_features = expansion_factor * input_dim
        self.k = k
        self.input_dim = input_dim
        self.n_features = n_features

        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0)

    def encode(self, x):
        x_centered = x - self.decoder.bias
        pre_act = self.encoder(x_centered)
        topk_vals, topk_idx = torch.topk(pre_act, self.k, dim=-1)
        sparse = torch.zeros_like(pre_act)
        sparse.scatter_(-1, topk_idx, torch.relu(topk_vals))
        return sparse

    def forward(self, x):
        sparse = self.encode(x)
        recon = self.decoder(sparse)
        return recon, sparse


def train_sae(hidden_states, input_dim, expansion_factor=4, k=None,
              lr=None, epochs=None, batch_size=None, device='cpu'):
    """
    Train SAE on frozen hidden state trajectories.

    Args:
        hidden_states: list of (T, hidden_dim) arrays from all trials
        input_dim: hidden state dimensionality
        expansion_factor: dictionary size multiplier (4x standard, 8x for rare features)
        k: sparsity level (active features per sample)

    Returns:
        Trained SAE, loss history
    """
    if k is None:
        k = SAE_K
    if lr is None:
        lr = SAE_LR
    if epochs is None:
        epochs = SAE_EPOCHS
    if batch_size is None:
        batch_size = SAE_BATCH_SIZE

    all_hidden = np.concatenate(hidden_states, axis=0)
    data = torch.tensor(all_hidden, dtype=torch.float32, device=device)
    N = data.shape[0]

    sae = SparseAutoencoder(input_dim, expansion_factor, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss, n_batches = 0.0, 0

        for i in range(0, N, batch_size):
            batch = data[perm[i:i+batch_size]]
            recon, sparse = sae(batch)
            recon_loss = nn.functional.mse_loss(recon, batch)
            norm_loss = ((torch.norm(sae.decoder.weight, dim=0) - 1.0)**2).mean()
            loss = recon_loss + 0.01 * norm_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += recon_loss.item()
            n_batches += 1

        loss_history.append(epoch_loss / n_batches)

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample = data[:min(10000, N)]
                feats = sae.encode(sample)
                alive = (feats > 0).any(dim=0).sum().item()
            logger.info(
                "  Epoch %d: loss=%.6f, alive=%d/%d",
                epoch + 1, loss_history[-1], alive, sae.n_features,
            )

    return sae, loss_history


def sae_probe_biological_variables(sae, hidden_states, bio_targets,
                                    target_names, device='cpu'):
    """
    Probe SAE features for biological variables.

    Two-stage: (1) Decompose hidden states into SAE features,
    (2) Ridge regression from SAE features to each biological variable.

    Also compute monosemanticity scores: does each SAE feature
    correspond to exactly one biological variable?

    Returns:
        Per-variable R2, per-feature correlation matrix,
        monosemanticity scores, and comparison to raw Ridge.
    """
    all_hidden = np.concatenate(hidden_states, axis=0)
    all_bio = np.concatenate(bio_targets, axis=0)

    # Decompose through SAE
    with torch.no_grad():
        h_tensor = torch.tensor(all_hidden, dtype=torch.float32, device=device)
        sae_features = sae.encode(h_tensor).cpu().numpy()

    n_features = sae_features.shape[1]
    n_bio = all_bio.shape[1] if all_bio.ndim > 1 else 1

    # Correlation matrix: (n_sae_features, n_bio_variables)
    corr_matrix = np.zeros((n_features, n_bio))
    for i in range(n_features):
        feat = sae_features[:, i]
        if feat.std() < 1e-10:
            continue  # Dead feature
        for j in range(n_bio):
            target = all_bio[:, j] if all_bio.ndim > 1 else all_bio
            if target.std() < 1e-10:
                continue
            corr_matrix[i, j] = np.corrcoef(feat, target)[0, 1]

    # Monosemanticity: normalized entropy of abs correlation per feature
    monosemanticity = np.zeros(n_features)
    for i in range(n_features):
        abs_corr = np.abs(corr_matrix[i, :])
        total = abs_corr.sum()
        if total < 1e-10:
            monosemanticity[i] = 0.0
            continue
        probs = abs_corr / total
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(n_bio)
        monosemanticity[i] = 1.0 - (entropy / max_entropy)

    # Ridge from SAE features (compare to raw Ridge)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    sae_r2 = {}
    raw_r2 = {}

    for j, name in enumerate(target_names):
        target = all_bio[:, j] if all_bio.ndim > 1 else all_bio

        # SAE features -> target
        sae_scores = []
        raw_scores = []
        for train_idx, test_idx in kf.split(sae_features):
            # SAE path
            ridge_sae = Ridge(alpha=1.0)
            ridge_sae.fit(sae_features[train_idx], target[train_idx])
            sae_scores.append(ridge_sae.score(sae_features[test_idx], target[test_idx]))

            # Raw path (for comparison)
            ridge_raw = Ridge(alpha=1.0)
            ridge_raw.fit(all_hidden[train_idx], target[train_idx])
            raw_scores.append(ridge_raw.score(all_hidden[test_idx], target[test_idx]))

        sae_r2[name] = np.mean(sae_scores)
        raw_r2[name] = np.mean(raw_scores)

    return {
        'correlation_matrix': corr_matrix,
        'monosemanticity_scores': monosemanticity,
        'sae_r2': sae_r2,
        'raw_r2': raw_r2,
        'n_alive': int((np.abs(corr_matrix).max(axis=1) > 0.01).sum()),
        'mean_monosemanticity': float(monosemanticity[monosemanticity > 0].mean()
                                       if (monosemanticity > 0).any() else 0.0),
        'superposition_detected': {
            name: sae_r2[name] > raw_r2[name] + 0.05
            for name in target_names
        },
    }
