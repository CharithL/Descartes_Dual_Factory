"""
L5PC DESCARTES -- Voltage-Only and Temporal Baselines

For Level B targets (effective conductances), a high Ridge R-squared from
hidden states is only meaningful if the network encodes MORE than what the
local compartment voltage already determines.

Since G_inf(V) = gbar * m_inf(V)^p * h_inf(V)^q, any conductance that is
nearly at steady state is trivially predicted from voltage alone.  The
network must exceed this voltage-only baseline to demonstrate genuine
multi-timescale representation of ion-channel kinetics.

Two baselines:
  1. Voltage-only:  RidgeCV from V_compartment -> G_eff
  2. Temporal:      Exponentially filtered voltage with channel-matched tau
                    (the simplest single-timescale model the network should beat)
"""

import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from l5pc.config import (
    CV_FOLDS,
    RIDGE_ALPHAS,
    DELTA_THRESHOLD_LEARNED,
    VOLTAGE_ABOVE_THRESHOLD,
    VOLTAGE_ABOVE_MODERATE,
    RECORDING_DT_MS,
)
from l5pc.utils.io import load_results_json, save_results_json
from l5pc.utils.metrics import exponential_filter

logger = logging.getLogger(__name__)


# Typical channel time constants (ms) at resting potential.
# Used to build the temporal baseline: an exponentially filtered voltage
# with tau matched to each channel's dominant kinetic timescale.
CHANNEL_TAUS = {
    'NaTa_t':   0.5,    # Fast sodium activation
    'Nap_Et2':  5.0,    # Persistent sodium
    'K_Tst':    7.0,    # Transient potassium
    'K_Pst':    10.0,   # Persistent potassium
    'SKv3_1':   4.0,    # Fast delayed rectifier
    'SK_E2':    50.0,   # Calcium-activated potassium (slow)
    'Im':       50.0,   # Muscarinic potassium (slow)
    'Ih':       100.0,  # HCN / h-current (very slow)
    'Ca_HVA':   5.0,    # High-voltage-activated calcium
    'Ca_LVAst': 20.0,   # Low-voltage-activated calcium
}


# ---------------------------------------------------------------------------
# Voltage-only baseline
# ---------------------------------------------------------------------------

def voltage_only_baseline(V_compartment, G_targets, cv_folds=CV_FOLDS,
                          alphas=None):
    """Fit RidgeCV from local compartment voltage alone to each G_eff.

    Parameters
    ----------
    V_compartment : ndarray, shape (n_trials,) or (n_trials, n_features)
        Trial-averaged voltage (or voltage features like mean, std, min, max)
        from the compartment where the conductance is located.
        If 1-D, expanded to column vector.
    G_targets : dict
        Mapping variable_name -> ndarray of shape (n_trials,).
    cv_folds : int
    alphas : list of float, optional

    Returns
    -------
    results : dict
        variable_name -> R2_voltage_only (float).
    """
    if alphas is None:
        alphas = RIDGE_ALPHAS

    # Ensure 2-D
    V = np.atleast_2d(V_compartment)
    if V.shape[0] == 1 and V.shape[1] != 1:
        V = V.T  # (n_trials, 1)
    if V.ndim == 1:
        V = V.reshape(-1, 1)

    results = {}
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for var_name, y in G_targets.items():
        n = min(len(V), len(y))
        X = V[:n]
        yy = y[:n]

        fold_r2s = []
        for train_idx, test_idx in kf.split(X):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            model = RidgeCV(alphas=alphas)
            model.fit(X_train, yy[train_idx])
            fold_r2s.append(float(model.score(X_test, yy[test_idx])))

        results[var_name] = float(np.mean(fold_r2s))

    return results


# ---------------------------------------------------------------------------
# Temporal baseline (exponentially filtered voltage)
# ---------------------------------------------------------------------------

def _build_voltage_features(V_traces, tau_ms, dt_ms=RECORDING_DT_MS):
    """Build feature matrix from raw voltage and its exponential filter.

    Parameters
    ----------
    V_traces : ndarray, shape (n_trials, T)
        Voltage traces for each trial.
    tau_ms : float
        Time constant in ms for the exponential filter.
    dt_ms : float

    Returns
    -------
    features : ndarray, shape (n_trials, 4)
        [mean_V, std_V, mean_filtered, std_filtered] per trial.
    """
    n_trials = V_traces.shape[0]
    features = np.zeros((n_trials, 4))

    for i in range(n_trials):
        v = V_traces[i]
        v_filt = exponential_filter(v, tau_ms, dt_ms)
        features[i, 0] = np.mean(v)
        features[i, 1] = np.std(v)
        features[i, 2] = np.mean(v_filt)
        features[i, 3] = np.std(v_filt)

    return features


def temporal_baseline(V_traces, G_targets, channel_taus=None,
                      dt_ms=RECORDING_DT_MS, cv_folds=CV_FOLDS,
                      alphas=None):
    """Exponentially filtered voltage baseline with channel-matched tau.

    For each conductance target, the feature set is:
      [mean_V, std_V, mean_V_filtered(tau), std_V_filtered(tau)]

    The network must exceed this to demonstrate it has learned a
    multi-timescale representation beyond simple low-pass filtering.

    Parameters
    ----------
    V_traces : ndarray, shape (n_trials, T)
        Raw voltage traces per trial.
    G_targets : dict
        variable_name -> ndarray (n_trials,).
    channel_taus : dict, optional
        Mapping channel_name -> tau_ms.  Defaults to CHANNEL_TAUS.
    dt_ms : float
    cv_folds : int
    alphas : list of float, optional

    Returns
    -------
    results : dict
        variable_name -> R2_temporal (float).
    """
    if channel_taus is None:
        channel_taus = CHANNEL_TAUS
    if alphas is None:
        alphas = RIDGE_ALPHAS

    results = {}
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for var_name, y in G_targets.items():
        # Extract channel name from variable name
        # Convention: "geff_NaTa_t_soma" -> channel = "NaTa_t"
        tau = _get_tau_for_variable(var_name, channel_taus)

        n = min(V_traces.shape[0], len(y))
        X = _build_voltage_features(V_traces[:n], tau, dt_ms)
        yy = y[:n]

        fold_r2s = []
        for train_idx, test_idx in kf.split(X):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            model = RidgeCV(alphas=alphas)
            model.fit(X_train, yy[train_idx])
            fold_r2s.append(float(model.score(X_test, yy[test_idx])))

        results[var_name] = float(np.mean(fold_r2s))

    return results


def _get_tau_for_variable(var_name, channel_taus):
    """Extract the appropriate tau for a variable name.

    Tries to match channel names in the variable name string.
    Falls back to 10.0 ms (moderate timescale) if no match.
    """
    for channel, tau in channel_taus.items():
        if channel in var_name:
            return tau
    logger.debug("No channel-tau match for '%s', using default 10.0 ms",
                 var_name)
    return 10.0


# ---------------------------------------------------------------------------
# Full baseline runner
# ---------------------------------------------------------------------------

def run_voltage_baselines(trial_dir, ridge_results_path, save_path,
                          hidden_dir=None):
    """Compute voltage baselines for all Level B variables with DeltaR2 > 0.1.

    For every qualifying variable:
      1. Compute R2_voltage_only from compartment voltage
      2. Compute R2_temporal from exponentially filtered voltage
      3. Compute R2_above_voltage = R2_trained - R2_voltage_only
      4. Reclassify per guide Section 4.7 interpretation thresholds

    Parameters
    ----------
    trial_dir : str or Path
        Directory with trial .npz files (containing voltage traces).
    ridge_results_path : str or Path
        Path to ridge results JSON for Level B.
    save_path : str or Path
        Output path for baseline results JSON.
    hidden_dir : str or Path, optional
        Not used directly (R2_trained comes from ridge results).

    Returns
    -------
    baseline_results : dict
    """
    trial_dir = Path(trial_dir)
    ridge_data = load_results_json(ridge_results_path)

    # Load voltage traces and conductance targets from trials
    V_traces, G_targets = _load_voltage_and_targets(trial_dir)

    # Filter to variables that passed the zombie threshold
    qualifying_vars = {}
    ridge_r2_map = {}
    for r in ridge_data.get('results', []):
        if r.get('delta_R2', 0) >= DELTA_THRESHOLD_LEARNED:
            vname = r['var_name']
            if vname in G_targets:
                qualifying_vars[vname] = G_targets[vname]
                ridge_r2_map[vname] = r['R2_trained']

    if not qualifying_vars:
        logger.info("No Level B variables above DeltaR2 threshold.")
        save_results_json({'results': [], 'n_qualifying': 0}, save_path)
        return {'results': [], 'n_qualifying': 0}

    logger.info("Computing voltage baselines for %d qualifying variables",
                len(qualifying_vars))

    # Build trial-averaged voltage features for voltage-only baseline
    V_mean = np.mean(V_traces, axis=1) if V_traces.ndim > 1 else V_traces
    voltage_r2 = voltage_only_baseline(V_mean, qualifying_vars)
    temporal_r2 = temporal_baseline(V_traces, qualifying_vars)

    # Assemble results with reclassification
    results = []
    for vname in qualifying_vars:
        r2_trained = ridge_r2_map[vname]
        r2_volt = voltage_r2.get(vname, 0.0)
        r2_temp = temporal_r2.get(vname, 0.0)
        r2_above = r2_trained - r2_volt

        # Reclassify (guide Section 4.7)
        if r2_above < VOLTAGE_ABOVE_MODERATE:
            baseline_category = 'VOLTAGE_REENCODING'
        elif r2_above < VOLTAGE_ABOVE_THRESHOLD:
            baseline_category = 'MARGINAL_ABOVE_VOLTAGE'
        else:
            baseline_category = 'ABOVE_VOLTAGE'

        entry = {
            'var_name': vname,
            'R2_trained': float(r2_trained),
            'R2_voltage_only': float(r2_volt),
            'R2_temporal': float(r2_temp),
            'R2_above_voltage': float(r2_above),
            'R2_above_temporal': float(r2_trained - r2_temp),
            'baseline_category': baseline_category,
        }
        results.append(entry)
        logger.info(
            "  %s: R2_tr=%.3f  R2_volt=%.3f  R2_above=%.3f  [%s]",
            vname, r2_trained, r2_volt, r2_above, baseline_category,
        )

    output = {
        'n_qualifying': len(results),
        'n_voltage_reencoding': sum(
            1 for r in results
            if r['baseline_category'] == 'VOLTAGE_REENCODING'
        ),
        'n_above_voltage': sum(
            1 for r in results
            if r['baseline_category'] == 'ABOVE_VOLTAGE'
        ),
        'results': results,
    }

    save_results_json(output, save_path)
    logger.info("Saved baseline results to %s", save_path)
    return output


def _load_voltage_and_targets(trial_dir):
    """Load voltage traces and Level B conductance targets from trial files.

    Returns
    -------
    V_traces : ndarray, shape (n_trials, T)
        Somatic voltage traces.
    G_targets : dict
        variable_name -> ndarray (n_trials,), trial-averaged conductances.
    """
    from l5pc.utils.io import load_all_trials, concat_trials

    trials = load_all_trials(trial_dir, split='all')
    if not trials:
        raise FileNotFoundError(f"No trial files found in {trial_dir}")

    # Voltage: use the output trace (somatic voltage is the training target)
    # shape per trial: (T,) or (T, n_compartments)
    V_list = []
    for t in trials:
        v = t.get('output', None)
        if v is not None:
            if v.ndim > 1:
                v = v[:, 0]  # First compartment = soma
            V_list.append(v)
    V_traces = np.array(V_list)  # (n_trials, T)

    # Level B conductance targets
    # Convention: level_B_cond has shape (T,) per trial for each variable
    # We take the trial-mean of each conductance over time
    G_targets = {}
    first_trial = trials[0]
    if 'level_B_cond' in first_trial:
        cond_data = first_trial['level_B_cond']
        # If structured array or 2-D, handle accordingly
        if cond_data.ndim == 1:
            # Single conductance variable -- use generic name
            G_targets['geff'] = np.array([
                np.mean(t['level_B_cond']) for t in trials
            ])
        elif cond_data.ndim == 2:
            n_vars = cond_data.shape[1] if cond_data.ndim == 2 else 1
            # Load variable names if available
            from l5pc.utils.io import load_variable_names
            var_names_meta = load_variable_names(trial_dir)
            if var_names_meta and 'level_B' in var_names_meta:
                names = var_names_meta['level_B']
            else:
                names = [f'geff_{i}' for i in range(n_vars)]
            for j, name in enumerate(names):
                G_targets[name] = np.array([
                    np.mean(t['level_B_cond'][:, j])
                    if t['level_B_cond'].ndim == 2
                    else np.mean(t['level_B_cond'])
                    for t in trials
                ])

    return V_traces, G_targets
