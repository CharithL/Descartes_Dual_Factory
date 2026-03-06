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

def ridge_cv_score(X, y, cv_folds=CV_FOLDS, alphas=None, target_name=None):
    """Fit RidgeCV with trial-level cross-validation.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_features)
    y : ndarray, shape (n_trials,)
    cv_folds : int
    alphas : list of float, optional
    target_name : str, optional
        Name of the target variable (for diagnostic logging).

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

    # Check for degenerate targets (zero variance)
    y_std = np.std(y)
    if y_std < 1e-10:
        label = target_name or "unknown"
        logger.warning("Target '%s' has zero variance (std=%.2e) -- returning R2=0",
                       label, y_std)
        return 0.0, [0.0] * cv_folds, alphas[0]

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_r2s = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # StandardScaler fitted per fold to avoid leakage
        scaler_x = StandardScaler()
        X_train = scaler_x.fit_transform(X_train)
        X_test = scaler_x.transform(X_test)

        # Standardize target y per fold (improves numerical stability,
        # especially for voltage targets in [-80, +40] mV range)
        y_mean, y_std_fold = y_train.mean(), y_train.std()
        if y_std_fold < 1e-10:
            fold_r2s.append(0.0)
            continue
        y_train_s = (y_train - y_mean) / y_std_fold
        y_test_s = (y_test - y_mean) / y_std_fold

        model = RidgeCV(alphas=alphas)
        model.fit(X_train, y_train_s)
        r2 = model.score(X_test, y_test_s)
        fold_r2s.append(float(r2))

    # Fit once on all data to report best alpha
    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X)
    y_full_s = (y - y.mean()) / max(y.std(), 1e-10)
    model_full = RidgeCV(alphas=alphas)
    model_full.fit(X_full, y_full_s)

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
                                                    alphas=alphas,
                                                    target_name=var_name)
        r2_un, folds_un, alpha_un = ridge_cv_score(X_un, target_y,
                                                    cv_folds=cv_folds,
                                                    alphas=alphas,
                                                    target_name=var_name)

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

    Supports two file conventions (tried in order):
        1. {hidden_dir}/lstm_{hidden_size}_{tag}.npz  (from extract_hidden.py)
        2. {hidden_dir}/h{hidden_size}_{tag}.npy       (legacy flat array)

    If the loaded array has shape (n_trials * T_STEPS, hidden_dim), it is
    automatically reshaped to (n_trials, T_STEPS, hidden_dim) and averaged
    over the time axis to produce (n_trials, hidden_dim).

    Shape returned: (n_trials, hidden_size)
    """
    from l5pc.config import T_STEPS, TEST_SPLIT

    tag = 'trained' if trained else 'untrained'
    hidden_dir = Path(hidden_dir)

    candidates = [
        hidden_dir / f'lstm_{hidden_size}_{tag}.npz',
        hidden_dir / f'h{hidden_size}_{tag}.npz',
        hidden_dir / f'h{hidden_size}_{tag}.npy',
    ]

    path = None
    for c in candidates:
        if c.exists():
            path = c
            break

    if path is None:
        raise FileNotFoundError(
            f"Hidden states not found for h={hidden_size} ({tag}). "
            f"Searched: {[str(c) for c in candidates]}"
        )

    # Load array depending on file format
    if path.suffix == '.npz':
        data = np.load(path)
        if 'hidden_states' in data:
            H = data['hidden_states']
        else:
            H = data[data.files[0]]
    else:
        H = np.load(path)

    # If timestep-level data, reshape and trial-average
    # Expected timestep shape: (n_trials * T_STEPS, hidden_dim)
    # Target shape:            (n_trials, hidden_dim)
    n_total = H.shape[0]
    if n_total > TEST_SPLIT * 2:
        n_trials = n_total // T_STEPS
        if n_trials * T_STEPS == n_total and n_trials > 0:
            logger.info(
                "Trial-averaging hidden states: (%d, %d) -> (%d, %d)",
                n_total, H.shape[1], n_trials, H.shape[1],
            )
            H = H.reshape(n_trials, T_STEPS, -1).mean(axis=1)

    return H


def _load_targets(targets_dir, level):
    """Load target variables for a given level.

    Tries three approaches in order:
        1. Aggregate file: {targets_dir}/{level_key}.npz
        2. Alternate name:  {targets_dir}/level_{level}_targets.npz
        3. Individual trial files (test split): loads each trial_{i}.npz
           and computes the trial-mean of each variable over time.

    Returns
    -------
    targets : dict
        Mapping variable_name -> ndarray of shape (n_trials,).
    """
    key = _LEVEL_KEYS[level]
    targets_dir = Path(targets_dir)

    # --- Approach 1 & 2: aggregate files ---
    for fname in [f'{key}.npz', f'level_{level}_targets.npz']:
        path = targets_dir / fname
        if path.exists():
            data = np.load(path)
            return {k: data[k] for k in data.files}

    # --- Approach 3: reconstruct from individual trial files ---
    from l5pc.utils.io import load_all_trials, load_variable_names

    trials = load_all_trials(targets_dir, split='test')
    if not trials:
        raise FileNotFoundError(
            f"No target data found for level {level} in {targets_dir}"
        )

    first_trial = trials[0]
    if key not in first_trial:
        raise FileNotFoundError(
            f"Level key '{key}' not found in trial data. "
            f"Available keys: {list(first_trial.keys())}"
        )

    arr0 = first_trial[key]
    var_meta = load_variable_names(targets_dir)

    if arr0.ndim == 1:
        # 1D array: could be EITHER:
        #   a) A time-series of length T for a single variable (Levels A/B)
        #   b) A vector of N scalar properties (Level C emergent)
        #
        # Distinguish by checking: does the metadata list multiple Level C
        # property names matching this array length?

        level_c_names = None
        if level == 'C' and var_meta:
            # Check 'level_C_keys' (saved by run_bahl_sim) or 'level_C'
            for mk in ['level_C_keys', 'level_C']:
                candidate = var_meta.get(mk)
                if (isinstance(candidate, list)
                        and len(candidate) > 1
                        and len(candidate) == len(arr0)):
                    level_c_names = candidate
                    break

        if level_c_names is not None:
            # Level C: vector of N scalar properties per trial
            # Each element is already a scalar -- no time-averaging needed
            targets = {}
            for j, name in enumerate(level_c_names):
                values = np.array([float(t[key][j]) for t in trials])
                targets[name] = values
            logger.info(
                "Level C: loaded %d emergent properties as separate targets",
                len(targets),
            )
            return targets
        else:
            # Single time-series variable: (T,) -> trial-mean scalar
            var_name = key
            if var_meta:
                for mk in var_meta:
                    if level.upper() in mk.upper() or level.lower() in mk.lower():
                        names = var_meta[mk]
                        if isinstance(names, list) and len(names) == 1:
                            var_name = names[0]
                        break
            values = np.array([float(np.mean(t[key])) for t in trials])
            return {var_name: values}

    elif arr0.ndim == 2:
        # Multiple variables: (T, n_vars) per trial -> trial-mean per var
        n_vars = arr0.shape[1]

        # Resolve variable names.
        # Prefer '_keys' metadata (matches npz column order from
        # run_bahl_sim's sorted dict keys) over generic var_names.
        names = None
        if var_meta:
            # Try specific key-order metadata first (matches npz column order)
            key_suffix = key.split('_', 1)[1] if '_' in key else key
            for mk in [f'{key}_keys', f'level_{level}_keys',
                        key, f'level_{level}']:
                candidate = var_meta.get(mk)
                if isinstance(candidate, list) and len(candidate) == n_vars:
                    names = candidate
                    break
            # Fallback: scan all metadata keys for a match
            if names is None:
                for mk in var_meta:
                    if level.upper() in mk.upper() or level.lower() in mk.lower():
                        candidate = var_meta[mk]
                        if isinstance(candidate, list) and len(candidate) == n_vars:
                            names = candidate
                            break
        if names is None:
            names = [f'{key}_{i}' for i in range(n_vars)]

        targets = {}
        for j, name in enumerate(names):
            values = np.array([
                float(np.mean(t[key][:, j]))
                if t[key].ndim == 2 and t[key].shape[1] > j
                else float(np.mean(t[key]))
                for t in trials
            ])
            targets[name] = values
        return targets

    else:
        raise FileNotFoundError(
            f"Unexpected shape for level {level} data: {arr0.shape}"
        )


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
