"""
L5PC DESCARTES -- Frequency-domain probing validation.

Methods 6, 8 from the 13-method suite:
  6. Frequency-resolved R2 (bandpass decomposition)
  8. Partial coherence conditioning (remove shared input drive)
"""

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold, cross_val_score


def frequency_resolved_r2(hidden, target, fs=1000, bands=None):
    """
    Decompose R2 into frequency bands.
    If R2 is driven entirely by <1 Hz band -> suspicious (shared drift).
    If R2 survives in 10-100 Hz -> genuine fast dynamics encoding.
    """
    if bands is None:
        bands = {
            'ultra_slow': (0.1, 1),
            'slow': (1, 10),
            'medium': (10, 100),
            'fast': (100, min(450, fs/2 - 1)),
        }

    results = {}
    ridge = Ridge(alpha=1.0)

    for band_name, (low, high) in bands.items():
        try:
            b, a = butter(4, [low, high], btype='band', fs=fs)
            target_filtered = filtfilt(b, a, target, axis=0)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for train_idx, test_idx in kf.split(hidden):
                ridge.fit(hidden[train_idx], target_filtered[train_idx])
                scores.append(
                    ridge.score(hidden[test_idx], target_filtered[test_idx]))

            results[band_name] = np.mean(scores)
        except Exception:
            results[band_name] = np.nan

    return results


def partial_coherence_r2(hidden, target, input_signal):
    """
    Remove shared input drive before probing.
    Regress out input from both hidden and target,
    then probe residuals.
    """
    # Regress out input from hidden states
    reg_h = LinearRegression().fit(input_signal, hidden)
    hidden_residual = hidden - reg_h.predict(input_signal)

    # Regress out input from target
    reg_t = LinearRegression().fit(input_signal, target.reshape(-1, 1))
    target_residual = target - reg_t.predict(input_signal).ravel()

    # Probe residuals
    ridge = Ridge(alpha=1.0)
    scores = cross_val_score(ridge, hidden_residual, target_residual,
                              cv=5, scoring='r2')

    return float(np.mean(scores))
