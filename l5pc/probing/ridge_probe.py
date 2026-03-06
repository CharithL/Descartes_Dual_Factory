"""
L5PC DESCARTES -- Ridge DeltaR-squared Probing Layer

Core probing methodology identical to the hippocampal bottleneck_ridge_all.py:
for each biophysical target variable, fit RidgeCV from trained hidden states
and from untrained (random-init) hidden states, then compute
DeltaR2 = R2_trained - R2_untrained.

Variables with DeltaR2 < DELTA_THRESHOLD_LEARNED are classified ZOMBIE:
the network carries no more information about them than random projections.

Levels:
  A -- individual gate variables (m, h) per channel per compartment
  B -- effective conductances g_eff = gbar * prod(gate^exp)
  C -- emergent properties (burst ratio, dendritic Ca integral, BAC flag, ...)
"""

import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from l5pc.config import (
    ABLATION_K_FRACTIONS,
    CV_FOLDS,
    DELTA_THRESHOLD_LEARNED,
    HIDDEN_SIZES,
    PREPROCESSING_OPTIONS,
    RIDGE_ALPHAS,
    SELECTIVITY_PERMS,
    P_THRESHOLD,
)
from l5pc.utils.io import load_results_json, save_results_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(X, method):
    """Apply preprocessing to hidden-state matrix.

    Parameters
    ----------
    X : ndarray, shape (n_trials, hidden_size) or (n_trials, n_features)
        Hidden-state activations averaged over time within each trial.
    method : str
        One of 'Raw', 'StandardScaler', 'PCA_5', 'PCA_10', 'PCA_20', 'PCA_50'.

    Returns
    -------
    X_proc : ndarray
        Preprocessed features, ready for Ridge regression.
    """
    if method == 'Raw':
        return X.copy()

    if method == 'StandardScaler':
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    if method.startswith('PCA_'):
        n_components = int(method.split('_')[1])
        n_components = min(n_components, X.shape[0], X.shape[1])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components, random_state=0)
        return pca.fit_transform(X_scaled)

    raise ValueError(f"Unknown preprocessing method: {method}")


# ---------------------------------------------------------------------------
# Ridge cross-validation
# ---------------------------------------------------------------------------

def ridge_cv_score(X, y, cv_folds=CV_FOLDS, alphas=None):
    """Fit RidgeCV with trial-level cross-validation.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_features)
    y : ndarray, shape (n_trials,)
    cv_folds : int
    alphas : list of float, optional

    Returns
    -------
    mean_r2 : float
        Mean R-squared across CV folds.
    fold_r2s : list of float
        Per-fold R-squared values.
    best_alpha : float
        Regularisation strength chosen by RidgeCV on the full dataset.
    """
    if alphas is None:
        alphas = RIDGE_ALPHAS

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_r2s = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # StandardScaler fitted per fold to avoid leakage
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RidgeCV(alphas=alphas)
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        fold_r2s.append(float(r2))

    # Fit once on all data to report best alpha
    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X)
    model_full = RidgeCV(alphas=alphas)
    model_full.fit(X_full, y)

    return float(np.mean(fold_r2s)), fold_r2s, float(model_full.alpha_)


# ---------------------------------------------------------------------------
# Permutation test for selectivity
# ---------------------------------------------------------------------------

def selectivity_permutation_test(X, y, n_perms=SELECTIVITY_PERMS,
                                 cv_folds=CV_FOLDS):
    """Permutation test: shuffle y across trials, refit, build null R2 dist.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_features)
    y : ndarray, shape (n_trials,)
    n_perms : int
    cv_folds : int

    Returns
    -------
    p_value : float
        Fraction of permuted R2 values >= observed R2.
    observed_r2 : float
    null_r2s : list of float
    """
    observed_r2, _, _ = ridge_cv_score(X, y, cv_folds=cv_folds)

    rng = np.random.RandomState(0)
    null_r2s = []
    for _ in range(n_perms):
        y_perm = rng.permutation(y)
        perm_r2, _, _ = ridge_cv_score(X, y_perm, cv_folds=cv_folds)
        null_r2s.append(perm_r2)

    # p-value: proportion of null >= observed (add 1 for continuity correction)
    p_value = float((np.sum(np.array(null_r2s) >= observed_r2) + 1)
                    / (n_perms + 1))
    return p_value, observed_r2, null_r2s


# ---------------------------------------------------------------------------
# Single-variable probe
# ---------------------------------------------------------------------------

def probe_single_variable(trained_H, untrained_H, target_y, var_name,
                          preprocessing_options=None, alphas=None,
                          cv_folds=CV_FOLDS):
    """Probe one target variable across all preprocessing pipelines.

    Parameters
    ----------
    trained_H : ndarray, shape (n_trials, hidden_size)
        Trial-averaged hidden states from the trained surrogate.
    untrained_H : ndarray, shape (n_trials, hidden_size)
        Trial-averaged hidden states from the untrained (random-init) model.
    target_y : ndarray, shape (n_trials,)
        Scalar biophysical target per trial.
    var_name : str
    preprocessing_options : list of str, optional
    alphas : list of float, optional
    cv_folds : int

    Returns
    -------
    result : dict
        Keys: var_name, R2_trained, R2_untrained, delta_R2,
              best_preprocessing, best_alpha, p_value, category,
              all_preprocessing_results.
    """
    if preprocessing_options is None:
        preprocessing_options = PREPROCESSING_OPTIONS
    if alphas is None:
        alphas = RIDGE_ALPHAS

    best_r2_trained = -np.inf
    best_r2_untrained = -np.inf
    best_prep = None
    all_prep_results = {}

    for prep in preprocessing_options:
        try:
            X_tr = preprocess(trained_H, prep)
            X_un = preprocess(untrained_H, prep)
        except Exception as e:
            logger.warning("Preprocessing '%s' failed for %s: %s",
                           prep, var_name, e)
            continue

        r2_tr, folds_tr, alpha_tr = ridge_cv_score(X_tr, target_y,
                                                    cv_folds=cv_folds,
                                                    alphas=alphas)
        r2_un, folds_un, alpha_un = ridge_cv_score(X_un, target_y,
                                                    cv_folds=cv_folds,
                                                    alphas=alphas)

        all_prep_results[prep] = {
            'R2_trained': r2_tr,
            'R2_untrained': r2_un,
            'delta_R2': r2_tr - r2_un,
            'alpha_trained': alpha_tr,
            'alpha_untrained': alpha_un,
            'fold_R2s_trained': folds_tr,
            'fold_R2s_untrained': folds_un,
        }

        if r2_tr > best_r2_trained:
            best_r2_trained = r2_tr
            best_r2_untrained = r2_un
            best_prep = prep

    delta_r2 = best_r2_trained - best_r2_untrained

    # Permutation test on best preprocessing
    p_value = 1.0
    if best_prep is not None and delta_r2 > 0:
        X_best = preprocess(trained_H, best_prep)
        p_value, _, _ = selectivity_permutation_test(
            X_best, target_y, n_perms=SELECTIVITY_PERMS, cv_folds=cv_folds
        )

    # Preliminary category assignment (refined later by baselines + ablation)
    if delta_r2 < DELTA_THRESHOLD_LEARNED or p_value > P_THRESHOLD:
        category = 'ZOMBIE'
    else:
        category = 'LEARNED'

    return {
        'var_name': var_name,
        'R2_trained': float(best_r2_trained),
        'R2_untrained': float(best_r2_untrained),
        'delta_R2': float(delta_r2),
        'best_preprocessing': best_prep,
        'p_value': float(p_value),
        'category': category,
        'all_preprocessing_results': all_prep_results,
    }


# ---------------------------------------------------------------------------
# Level-wide probing
# ---------------------------------------------------------------------------

_LEVEL_KEYS = {
    'A': 'level_A_gates',
    'B': 'level_B_cond',
    'C': 'level_C_emerge',
}


def _load_hidden_states(hidden_dir, hidden_size, trained=True):
    """Load trial-averaged hidden states.

    Expected file convention:
        {hidden_dir}/h{hidden_size}_{'trained'|'untrained'}.npy
    Shape: (n_trials, hidden_size)
    """
    tag = 'trained' if trained else 'untrained'
    path = Path(hidden_dir) / f'h{hidden_size}_{tag}.npy'
    if not path.exists():
        raise FileNotFoundError(f"Hidden states not found: {path}")
    return np.load(path)


def _load_targets(targets_dir, level):
    """Load target variables for a given level.

    Expected file convention:
        {targets_dir}/{level}_targets.npz
    Each key in the npz is a variable name, value shape (n_trials,).
    """
    key = _LEVEL_KEYS[level]
    path = Path(targets_dir) / f'{key}.npz'
    if not path.exists():
        # Fallback: try level letter
        path = Path(targets_dir) / f'level_{level}_targets.npz'
    if not path.exists():
        raise FileNotFoundError(
            f"Target file not found for level {level} in {targets_dir}"
        )
    data = np.load(path)
    return {k: data[k] for k in data.files}


def run_probe(level, hidden_size, hidden_dir, targets_dir, save_path):
    """Run Ridge DeltaR2 probing for one level and one hidden size.

    Parameters
    ----------
    level : str
        'A', 'B', or 'C'.
    hidden_size : int
        LSTM hidden dimension (64, 128, or 256).
    hidden_dir : str or Path
        Directory containing hidden-state .npy files.
    targets_dir : str or Path
        Directory containing target .npz files.
    save_path : str or Path
        Output JSON path.
    """
    logger.info("Probing level=%s  hidden_size=%d", level, hidden_size)

    trained_H = _load_hidden_states(hidden_dir, hidden_size, trained=True)
    untrained_H = _load_hidden_states(hidden_dir, hidden_size, trained=False)
    targets = _load_targets(targets_dir, level)

    n_trials_h = trained_H.shape[0]
    results = []

    for var_name, y in targets.items():
        # Ensure matching trial counts
        n_trials_y = len(y)
        n = min(n_trials_h, n_trials_y)
        if n < CV_FOLDS:
            logger.warning("Skipping %s: only %d trials (< %d folds)",
                           var_name, n, CV_FOLDS)
            continue

        result = probe_single_variable(
            trained_H[:n], untrained_H[:n], y[:n], var_name
        )
        result['level'] = level
        result['hidden_size'] = hidden_size
        results.append(result)

        logger.info(
            "  %s: R2_tr=%.3f  R2_un=%.3f  dR2=%.3f  [%s]",
            var_name, result['R2_trained'], result['R2_untrained'],
            result['delta_R2'], result['category'],
        )

    # Summary statistics
    n_zombie = sum(1 for r in results if r['category'] == 'ZOMBIE')
    n_learned = sum(1 for r in results if r['category'] == 'LEARNED')

    output = {
        'level': level,
        'hidden_size': hidden_size,
        'n_variables': len(results),
        'n_zombie': n_zombie,
        'n_learned': n_learned,
        'zombie_fraction': n_zombie / max(len(results), 1),
        'results': results,
    }

    save_results_json(output, save_path)
    logger.info("Saved ridge results to %s  (%d zombie / %d total)",
                save_path, n_zombie, len(results))
    return output


def run_all_probes(hidden_dir, targets_dir, results_dir):
    """Run probing for all levels and all hidden sizes.

    Parameters
    ----------
    hidden_dir : str or Path
    targets_dir : str or Path
    results_dir : str or Path

    Returns
    -------
    all_results : dict
        Nested dict: all_results[level][hidden_size] = result_dict
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for level in ['A', 'B', 'C']:
        all_results[level] = {}
        for hs in HIDDEN_SIZES:
            save_path = results_dir / f'ridge_level{level}_h{hs}.json'
            try:
                result = run_probe(level, hs, hidden_dir, targets_dir,
                                   save_path)
                all_results[level][hs] = result
            except FileNotFoundError as e:
                logger.warning("Skipping level=%s h=%d: %s", level, hs, e)
                continue

    # Cross-level summary
    summary_path = results_dir / 'ridge_summary.json'
    summary = {}
    for level in all_results:
        for hs in all_results[level]:
            key = f'{level}_h{hs}'
            r = all_results[level][hs]
            summary[key] = {
                'n_variables': r['n_variables'],
                'n_zombie': r['n_zombie'],
                'n_learned': r['n_learned'],
                'zombie_fraction': r['zombie_fraction'],
            }
    save_results_json(summary, summary_path)
    logger.info("Saved cross-level summary to %s", summary_path)

    return all_results
