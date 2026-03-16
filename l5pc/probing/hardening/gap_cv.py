"""
L5PC DESCARTES -- Temporal gap cross-validation and cluster permutation tests.

Methods 11, 12 from the 13-method suite:
  11. Gap temporal CV (prevents autocorrelation leakage)
  12. Cluster-based permutation testing (Maris & Oostenveld 2007)
"""

import numpy as np
from scipy.ndimage import label
from sklearn.linear_model import Ridge


def gap_temporal_cv(hidden, target, n_splits=5, gap_size=50):
    """
    Leave temporal gaps between train and test to prevent
    autocorrelation leakage.
    """
    T = len(target)
    fold_size = T // n_splits
    scores = []
    ridge = Ridge(alpha=1.0)

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = min(test_start + fold_size, T)

        # Training: everything NOT in [test_start - gap, test_end + gap]
        train_mask = np.ones(T, dtype=bool)
        train_mask[max(0, test_start - gap_size):min(T, test_end + gap_size)] = False

        if train_mask.sum() < 100:
            continue

        ridge.fit(hidden[train_mask], target[train_mask])
        scores.append(ridge.score(hidden[test_start:test_end],
                                   target[test_start:test_end]))

    return np.mean(scores) if scores else np.nan


def cluster_permutation_test(delta_r2_map, n_permutations=1000,
                              cluster_threshold=0.05, rng=None):
    """
    For spatially/temporally structured probing results.
    Forms clusters of adjacent significant results and tests
    cluster-level significance.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Find observed clusters above threshold
    observed_significant = delta_r2_map > cluster_threshold
    labeled, n_clusters = label(observed_significant)

    # Cluster mass = sum of delta-R2 within each cluster
    observed_masses = []
    for c in range(1, n_clusters + 1):
        mass = delta_r2_map[labeled == c].sum()
        observed_masses.append(mass)

    if not observed_masses:
        return {'n_clusters': 0, 'significant_clusters': []}

    max_observed = max(observed_masses)

    # Permutation distribution of max cluster mass
    null_max_masses = []
    for _ in range(n_permutations):
        # Sign-flip permutation
        signs = rng.choice([-1, 1], size=delta_r2_map.shape)
        perm_map = delta_r2_map * signs

        perm_sig = perm_map > cluster_threshold
        perm_labeled, perm_n = label(perm_sig)

        if perm_n > 0:
            perm_masses = [perm_map[perm_labeled == c].sum()
                          for c in range(1, perm_n + 1)]
            null_max_masses.append(max(perm_masses))
        else:
            null_max_masses.append(0)

    null_max_masses = np.array(null_max_masses)

    # P-values per cluster
    cluster_results = []
    for mass in observed_masses:
        p = (null_max_masses >= mass).mean()
        cluster_results.append({'mass': mass, 'p_value': p,
                                'significant': p < 0.05})

    return {'n_clusters': n_clusters, 'clusters': cluster_results}
