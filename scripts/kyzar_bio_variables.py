"""
kyzar/bio_variables.py

DESCARTES Circuit 6 Phase 1.2: Biological probe target variables.

Computes 18 candidate variables from raw spike data for probing
LSTM hidden states. These are the biological ground truth targets
that the surrogate may or may not encode.

All variables are computed per time bin, aligned to X and Y tensors.
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

log = logging.getLogger(__name__)


def compute_all_bio_variables(X_trial, Y_trial, epoch_mask, trial_meta,
                              dt_ms=10):
    """Compute all 18 biological variables for one trial.

    Args:
        X_trial: (T, n_limbic) smoothed input firing rates
        Y_trial: (T, n_frontal) smoothed output firing rates
        epoch_mask: (T,) epoch identity codes
        trial_meta: dict with load, accuracy, reaction_time_ms, etc.
        dt_ms: bin width in ms

    Returns:
        bio_targets: (T, n_variables) array
        variable_names: list of variable name strings
    """
    T = X_trial.shape[0]
    dt_s = dt_ms / 1000.0
    fs = 1.0 / dt_s

    variables = {}

    # --- RATE-BASED ---
    variables['firing_rate_input'] = np.mean(X_trial, axis=1)
    variables['firing_rate_output'] = np.mean(Y_trial, axis=1)

    win = max(1, int(200 / dt_ms))
    fr_all = np.concatenate([X_trial, Y_trial], axis=1)
    mean_fr = np.mean(fr_all, axis=1)
    variables['trial_variance'] = _rolling_var(mean_fr, win)
    variables['delay_stability'] = _running_autocorr(mean_fr, win)

    # --- OSCILLATORY ---
    pop_rate = np.mean(X_trial, axis=1)
    variables['theta_power'] = _bandpower(pop_rate, fs, 4, 8)
    variables['gamma_power'] = _bandpower(pop_rate, fs, 30, min(80, fs/2 - 1))
    variables['theta_phase'] = _instantaneous_phase(pop_rate, fs, 4, 8)
    variables['theta_gamma_pac'] = _pac_strength(pop_rate, fs)

    # --- SYNCHRONY ---
    if X_trial.shape[1] >= 2:
        variables['population_synchrony'] = _sliding_pairwise_corr(X_trial, win)
    else:
        variables['population_synchrony'] = np.zeros(T)

    out_pop = np.mean(Y_trial, axis=1)
    variables['cross_region_coherence'] = _sliding_coherence(
        pop_rate, out_pop, fs, win, fmin=4, fmax=8)

    # --- TASK ---
    variables['working_memory_load'] = np.full(T, trial_meta.get('load', 1),
                                               dtype=np.float64)
    variables['choice_signal'] = np.full(T, trial_meta.get('probe_in_out', 0),
                                         dtype=np.float64)
    variables['accuracy'] = np.full(T, trial_meta.get('accuracy', 0),
                                    dtype=np.float64)
    variables['reaction_time'] = np.full(
        T, trial_meta.get('reaction_time_ms', 0), dtype=np.float64)

    # --- TEMPORAL/EPOCH ---
    variables['temporal_position'] = np.linspace(0, 1, T)
    variables['epoch_identity'] = epoch_mask.astype(np.float64)

    fix_mask = epoch_mask == 0
    maint_mask = epoch_mask == 2
    if np.any(fix_mask) and np.any(maint_mask):
        baseline = np.mean(mean_fr[fix_mask])
        maint_activity = mean_fr - baseline
    else:
        maint_activity = mean_fr - np.mean(mean_fr)
    variables['maintenance_persistent'] = maint_activity

    # --- CONTENT ---
    variables['stimulus_category'] = np.full(
        T, trial_meta.get('enc1_pic', 0), dtype=np.float64)

    var_names = list(variables.keys())
    bio_targets = np.column_stack([variables[v] for v in var_names])

    return bio_targets, var_names


def compute_session_bio_targets(session_dir, dt_ms=10):
    """Compute bio targets for all trials in a preprocessed session.

    Args:
        session_dir: path containing X_trials.npz, Y_trials.npz,
                     epoch_masks.npz, metadata.json

    Saves bio_targets.npz and bio_variable_names.json to session_dir.
    """
    session_dir = Path(session_dir)

    X_data = dict(np.load(session_dir / 'X_trials.npz'))
    Y_data = dict(np.load(session_dir / 'Y_trials.npz'))
    E_data = dict(np.load(session_dir / 'epoch_masks.npz'))

    with open(session_dir / 'metadata.json') as f:
        metadata = json.load(f)

    n_trials = metadata['n_trials']
    trial_metas = metadata['trial_metadata']

    all_targets = {}
    var_names = None

    for ti in range(n_trials):
        key = f'trial_{ti}'
        X_trial = X_data[key]
        Y_trial = Y_data[key]
        epoch_mask = E_data[key]
        trial_meta = trial_metas[ti]

        bio, names = compute_all_bio_variables(
            X_trial, Y_trial, epoch_mask, trial_meta, dt_ms)

        all_targets[key] = bio
        if var_names is None:
            var_names = names

    np.savez(session_dir / 'bio_targets.npz', **all_targets)

    with open(session_dir / 'bio_variable_names.json', 'w') as f:
        json.dump(var_names, f, indent=2)

    log.info("Saved %d trials x %d variables to %s",
             n_trials, len(var_names), session_dir)

    return var_names


# =====================================================================
# Helper functions for signal processing
# =====================================================================

def _rolling_var(x, win):
    """Rolling variance with window size win."""
    T = len(x)
    result = np.zeros(T)
    half = win // 2
    for i in range(T):
        lo = max(0, i - half)
        hi = min(T, i + half + 1)
        result[i] = np.var(x[lo:hi]) if hi - lo > 1 else 0
    return result


def _running_autocorr(x, win):
    """Running lag-1 autocorrelation in sliding windows."""
    T = len(x)
    result = np.zeros(T)
    half = win // 2
    for i in range(T):
        lo = max(0, i - half)
        hi = min(T, i + half + 1)
        seg = x[lo:hi]
        if len(seg) > 2:
            r = np.corrcoef(seg[:-1], seg[1:])[0, 1]
            result[i] = r if np.isfinite(r) else 0
    return result


def _bandpower(x, fs, fmin, fmax):
    """Instantaneous band power via bandpass then Hilbert transform."""
    T = len(x)
    if T < 20 or fs < 2 * fmax:
        return np.zeros(T)
    try:
        nyq = fs / 2
        low = max(fmin / nyq, 0.01)
        high = min(fmax / nyq, 0.99)
        if low >= high:
            return np.zeros(T)
        b, a = butter(3, [low, high], btype='band')
        padlen = min(3 * max(len(b), len(a)), T - 1)
        filtered = filtfilt(b, a, x, padlen=padlen)
        analytic = hilbert(filtered)
        return np.abs(analytic) ** 2
    except Exception:
        return np.zeros(T)


def _instantaneous_phase(x, fs, fmin, fmax):
    """Instantaneous phase in a frequency band."""
    T = len(x)
    if T < 20 or fs < 2 * fmax:
        return np.zeros(T)
    try:
        nyq = fs / 2
        low = max(fmin / nyq, 0.01)
        high = min(fmax / nyq, 0.99)
        if low >= high:
            return np.zeros(T)
        b, a = butter(3, [low, high], btype='band')
        padlen = min(3 * max(len(b), len(a)), T - 1)
        filtered = filtfilt(b, a, x, padlen=padlen)
        analytic = hilbert(filtered)
        return np.angle(analytic)
    except Exception:
        return np.zeros(T)


def _pac_strength(x, fs, theta_range=(4, 8), gamma_range=(30, 80)):
    """Theta-gamma phase-amplitude coupling via mean vector length."""
    T = len(x)
    gmax = min(gamma_range[1], fs / 2 - 1)
    if T < 50 or fs < 2 * gmax:
        return np.zeros(T)
    try:
        theta_phase = _instantaneous_phase(x, fs, *theta_range)
        gamma_amp = np.sqrt(np.maximum(
            _bandpower(x, fs, gamma_range[0], gmax), 0))
        win = max(1, int(500 / (1000 / fs)))
        half = win // 2
        pac = np.zeros(T)
        for i in range(T):
            lo = max(0, i - half)
            hi = min(T, i + half + 1)
            if hi - lo < 5:
                continue
            z = np.mean(gamma_amp[lo:hi] * np.exp(1j * theta_phase[lo:hi]))
            pac[i] = np.abs(z)
        return pac
    except Exception:
        return np.zeros(T)


def _sliding_pairwise_corr(X, win):
    """Mean pairwise |correlation| in sliding windows."""
    T, N = X.shape
    result = np.zeros(T)
    half = win // 2
    if N < 2:
        return result
    for i in range(T):
        lo = max(0, i - half)
        hi = min(T, i + half + 1)
        if hi - lo < 3:
            continue
        seg = X[lo:hi]
        try:
            corr_mat = np.corrcoef(seg.T)
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            vals = corr_mat[mask]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                result[i] = np.mean(np.abs(vals))
        except Exception:
            pass
    return result


def _sliding_coherence(x, y, fs, win, fmin=4, fmax=8):
    """Sliding coherence proxy: |correlation of bandpassed signals|."""
    T = len(x)
    if T < 20:
        return np.zeros(T)
    try:
        nyq = fs / 2
        low = max(fmin / nyq, 0.01)
        high = min(fmax / nyq, 0.99)
        if low >= high:
            return np.zeros(T)
        b, a = butter(3, [low, high], btype='band')
        padlen = min(3 * max(len(b), len(a)), T - 1)
        x_filt = filtfilt(b, a, x, padlen=padlen)
        y_filt = filtfilt(b, a, y, padlen=padlen)
        half = win // 2
        result = np.zeros(T)
        for i in range(T):
            lo = max(0, i - half)
            hi = min(T, i + half + 1)
            if hi - lo < 3:
                continue
            r = np.corrcoef(x_filt[lo:hi], y_filt[lo:hi])[0, 1]
            result[i] = abs(r) if np.isfinite(r) else 0
        return result
    except Exception:
        return np.zeros(T)
