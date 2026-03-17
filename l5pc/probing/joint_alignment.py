"""
L5PC DESCARTES -- Tier 2: Joint Alignment Probes

CCA, RSA, CKA, pi-VAE, CEBRA — tests whether hidden states share
representational geometry with biological variables.

CCA: shared low-dimensional subspace
RSA: pairwise distance structure match
CKA: nonlinear kernel alignment (Kornblith et al. 2019)
pi-VAE: identifiable conditional latent recovery (Zhou & Wei 2020)
CEBRA: joint neural-behavioral embedding (Schneider et al. 2023)
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA as sklearn_CCA
from sklearn.model_selection import KFold

from l5pc.probing.hardening.permutation import block_permute
from l5pc.probing.registry import is_available

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 6.1  CCA (Canonical Correlation Analysis)
# ──────────────────────────────────────────────────────────────

def cca_alignment(hidden, bio_targets, n_components=10, n_permutations=200,
                   block_size=50, seed=42):
    """
    Cross-validated CCA between hidden states and biological variables.

    Returns canonical correlations and permutation-tested significance.
    """
    n_components = min(n_components, hidden.shape[1], bio_targets.shape[1])
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    # Real CCA
    real_ccs = []
    for train_idx, test_idx in kf.split(hidden):
        cca = sklearn_CCA(n_components=n_components)
        cca.fit(hidden[train_idx], bio_targets[train_idx])
        X_c, Y_c = cca.transform(hidden[test_idx], bio_targets[test_idx])

        fold_ccs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
                    for i in range(n_components)]
        real_ccs.append(fold_ccs)

    real_ccs = np.mean(real_ccs, axis=0)

    # Block-permutation null
    rng = np.random.default_rng(seed)
    null_ccs = []
    for _ in range(n_permutations):
        bio_perm = block_permute(bio_targets, block_size, rng)

        perm_scores = []
        for train_idx, test_idx in kf.split(hidden):
            cca = sklearn_CCA(n_components=n_components)
            cca.fit(hidden[train_idx], bio_perm[train_idx])
            X_c, Y_c = cca.transform(hidden[test_idx], bio_perm[test_idx])
            perm_scores.append(
                [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
                 for i in range(n_components)])
        null_ccs.append(np.mean(perm_scores, axis=0))

    null_ccs = np.array(null_ccs)
    p_values = [(null_ccs[:, i] >= real_ccs[i]).mean()
                for i in range(n_components)]

    return {
        'canonical_correlations': real_ccs.tolist(),
        'p_values': p_values,
        'n_significant': sum(p < 0.05 for p in p_values),
        'mean_cc': float(real_ccs.mean()),
        'max_cc': float(real_ccs.max()),
    }


# ──────────────────────────────────────────────────────────────
# 6.2  RSA (Representational Similarity Analysis)
# ──────────────────────────────────────────────────────────────

def rsa_comparison(hidden, bio_targets, n_samples=2000, seed=42):
    """
    RSA: compare pairwise distance structures between
    hidden states and biological variables.
    """
    rng = np.random.default_rng(seed)

    # Subsample for computational tractability
    if len(hidden) > n_samples:
        idx = rng.choice(len(hidden), n_samples, replace=False)
        hidden = hidden[idx]
        bio_targets = bio_targets[idx]

    # Representational dissimilarity matrices
    rdm_hidden = pdist(hidden, metric='correlation')
    rdm_bio = pdist(bio_targets, metric='correlation')

    # Spearman correlation between RDMs
    rho, p_val = spearmanr(rdm_hidden, rdm_bio)

    return {
        'rsa_correlation': float(rho),
        'p_value': float(p_val),
        'geometric_match': rho > 0.3 and p_val < 0.001
    }


# ──────────────────────────────────────────────────────────────
# 6.3  CKA (Centered Kernel Alignment)
# ──────────────────────────────────────────────────────────────

def cka_comparison(hidden, bio_targets, kernel='rbf', sigma=None):
    """
    Centered Kernel Alignment (Kornblith et al. 2019).
    Invariant to rotation and isotropic scaling.
    """
    def centering_matrix(n):
        return np.eye(n) - np.ones((n, n)) / n

    def rbf_kernel(X, sigma):
        sq_dists = pdist(X, 'sqeuclidean')
        K = squareform(np.exp(-sq_dists / (2 * sigma**2)))
        np.fill_diagonal(K, 1.0)
        return K

    def linear_kernel(X):
        return X @ X.T

    n = len(hidden)
    H = centering_matrix(n)

    if kernel == 'rbf':
        if sigma is None:
            sigma_h = np.median(pdist(hidden))
            sigma_b = np.median(pdist(bio_targets))
        else:
            sigma_h = sigma_b = sigma
        K = rbf_kernel(hidden, sigma_h)
        L = rbf_kernel(bio_targets, sigma_b)
    else:
        K = linear_kernel(hidden)
        L = linear_kernel(bio_targets)

    HSIC_KL = np.trace(K @ H @ L @ H) / (n - 1)**2
    HSIC_KK = np.trace(K @ H @ K @ H) / (n - 1)**2
    HSIC_LL = np.trace(L @ H @ L @ H) / (n - 1)**2

    cka = HSIC_KL / (np.sqrt(HSIC_KK * HSIC_LL) + 1e-10)

    return {'cka': float(cka), 'kernel': kernel}


# ──────────────────────────────────────────────────────────────
# 6.4  pi-VAE (Identifiable Conditional Latent Recovery)
# ──────────────────────────────────────────────────────────────

class PiVAE(nn.Module):
    """
    pi-VAE (Zhou & Wei 2020): identifiable latent recovery
    conditioned on stimulus/task labels.

    Key: conditioning on stimulus identity breaks the rotation
    symmetry that makes standard VAE latents unidentifiable.
    """

    def __init__(self, input_dim, latent_dim, condition_dim, hidden_dim=128):
        super().__init__()

        # Encoder: h -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z + condition -> h_reconstructed
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Conditional prior: condition -> prior (mu, logvar)
        self.prior_mu = nn.Linear(condition_dim, latent_dim)
        self.prior_logvar = nn.Linear(condition_dim, latent_dim)

    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=-1))
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)

        prior_mu = self.prior_mu(c)
        prior_logvar = self.prior_logvar(c)

        return recon, mu, logvar, prior_mu, prior_logvar, z


# ──────────────────────────────────────────────────────────────
# 6.5  CEBRA (Joint Neural-Behavioral Embedding)
# ──────────────────────────────────────────────────────────────

def cebra_alignment(hidden, bio_targets, output_dim=8,
                     max_iterations=5000):
    """
    CEBRA (Schneider, Lee & Mathis 2023, Nature):
    Joint neural-behavioral embedding enabling direct
    latent-space comparison between LSTM and biology.

    Requires: pip install cebra
    """
    if not is_available('cebra'):
        return {'error': 'cebra not installed: pip install cebra'}

    try:
        import cebra

        model = cebra.CEBRA(
            model_architecture='offset10-model',
            batch_size=512,
            learning_rate=3e-4,
            max_iterations=max_iterations,
            output_dimension=output_dim,
        )

        model.fit(hidden, bio_targets)
        embedding = model.transform(hidden)

        # Measure alignment: decode bio from CEBRA embedding
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        r2_scores = {}
        for j in range(bio_targets.shape[1]):
            scores = cross_val_score(
                Ridge(1.0), embedding, bio_targets[:, j], cv=5)
            r2_scores[f'var_{j}'] = float(np.mean(scores))

        return {
            'embedding_shape': embedding.shape,
            'bio_recovery_r2': r2_scores,
            'mean_r2': float(np.mean(list(r2_scores.values())))
        }
    except ImportError:
        return {'error': 'cebra not installed: pip install cebra'}
