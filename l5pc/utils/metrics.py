"""Metrics for surrogate evaluation and validation."""
import numpy as np
from scipy import stats, signal


def cross_condition_correlation(predicted_rates, true_rates):
    """Cross-condition correlation: correlate MEAN rates across conditions.

    This is the primary output metric — NOT per-trial temporal correlation.
    The hippocampal experiment showed per-trial temporal correlation can be
    near zero even when the model performs well.
    """
    if predicted_rates.ndim == 2:
        pred_mean = predicted_rates.mean(axis=1)
        true_mean = true_rates.mean(axis=1)
    else:
        pred_mean = predicted_rates
        true_mean = true_rates
    if len(pred_mean) < 3:
        return 0.0
    r, _ = stats.pearsonr(pred_mean, true_mean)
    return float(r)


def detect_spikes(voltage, threshold=0.0, dt_ms=0.5):
    """Detect spike times from voltage trace.

    Returns:
        Array of spike times in ms
    """
    above = voltage > threshold
    crossings = np.where(np.diff(above.astype(int)) == 1)[0]
    return crossings * dt_ms


def isi_distribution(spike_times):
    """Compute inter-spike intervals."""
    if len(spike_times) < 2:
        return np.array([])
    return np.diff(spike_times)


def burst_ratio(spike_times, burst_isi_ms=10.0):
    """Fraction of spikes occurring in bursts (ISI < threshold)."""
    isis = isi_distribution(spike_times)
    if len(isis) == 0:
        return 0.0
    return float(np.sum(isis < burst_isi_ms)) / len(isis)


def ks_test_isi(spike_times_a, spike_times_b):
    """Kolmogorov-Smirnov test on ISI distributions."""
    isis_a = isi_distribution(spike_times_a)
    isis_b = isi_distribution(spike_times_b)
    if len(isis_a) < 2 or len(isis_b) < 2:
        return 1.0, 1.0  # No meaningful comparison
    stat, p = stats.ks_2samp(isis_a, isis_b)
    return float(stat), float(p)


def victor_purpura_distance(spikes_a, spikes_b, q=1.0):
    """Victor-Purpura spike distance with cost parameter q.

    Measures dissimilarity between two spike trains.
    q controls the time-precision vs spike-count trade-off.
    """
    n_a, n_b = len(spikes_a), len(spikes_b)
    if n_a == 0:
        return float(n_b)
    if n_b == 0:
        return float(n_a)

    D = np.zeros((n_a + 1, n_b + 1))
    D[:, 0] = np.arange(n_a + 1)
    D[0, :] = np.arange(n_b + 1)

    for i in range(1, n_a + 1):
        for j in range(1, n_b + 1):
            cost = q * abs(spikes_a[i - 1] - spikes_b[j - 1])
            D[i, j] = min(
                D[i - 1, j] + 1,          # Delete spike from a
                D[i, j - 1] + 1,          # Insert spike from b
                D[i - 1, j - 1] + cost    # Move spike
            )
    return float(D[n_a, n_b])


def psd_band_power(signal_data, fs, band):
    """Compute power spectral density in a frequency band.

    Args:
        signal_data: 1D signal
        fs: Sampling frequency (Hz)
        band: Tuple (low_hz, high_hz)

    Returns:
        Total power in the band
    """
    nperseg = min(256, len(signal_data))
    if nperseg < 16:
        return 0.0
    freqs, pxx = signal.welch(signal_data, fs=fs, nperseg=nperseg)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0
    return float(np.trapz(pxx[mask], freqs[mask]))


def fano_factor(spike_counts):
    """Fano factor: variance / mean of spike counts."""
    mean_c = np.mean(spike_counts)
    if mean_c == 0:
        return 0.0
    return float(np.var(spike_counts) / mean_c)


def exponential_filter(signal_data, tau_ms, dt_ms=0.5):
    """Apply causal exponential filter with time constant tau.

    Used for temporal baseline in voltage-only controls.
    """
    alpha = dt_ms / tau_ms
    filtered = np.zeros_like(signal_data)
    filtered[0] = signal_data[0]
    for t in range(1, len(signal_data)):
        filtered[t] = alpha * signal_data[t] + (1 - alpha) * filtered[t - 1]
    return filtered
