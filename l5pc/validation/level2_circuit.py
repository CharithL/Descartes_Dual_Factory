"""Level 2 Validation: Circuit Integration.

Measures whether the surrogate preserves network-level dynamics:
oscillations, correlations, variability.
"""
import numpy as np
from l5pc.utils.metrics import psd_band_power, fano_factor
from l5pc.config import GAMMA_BAND, BETA_BAND


def evaluate_oscillation_power(lfp_bio, lfp_surrogate, fs_hz=2000.0):
    """Compare gamma and beta oscillation power.

    LFP approximated as sum of synaptic currents or membrane potentials.

    Args:
        lfp_bio: 1D array, biological circuit LFP
        lfp_surrogate: 1D array, hybrid circuit LFP
        fs_hz: Sampling frequency
    """
    gamma_bio = psd_band_power(lfp_bio, fs_hz, GAMMA_BAND)
    gamma_surr = psd_band_power(lfp_surrogate, fs_hz, GAMMA_BAND)
    beta_bio = psd_band_power(lfp_bio, fs_hz, BETA_BAND)
    beta_surr = psd_band_power(lfp_surrogate, fs_hz, BETA_BAND)

    return {
        'gamma_power_bio': gamma_bio,
        'gamma_power_surrogate': gamma_surr,
        'gamma_ratio': gamma_surr / gamma_bio if gamma_bio > 0 else float('inf'),
        'beta_power_bio': beta_bio,
        'beta_power_surrogate': beta_surr,
        'beta_ratio': beta_surr / beta_bio if beta_bio > 0 else float('inf'),
    }


def evaluate_pairwise_correlations(spike_trains_bio, spike_trains_surrogate, bin_ms=5.0, dt_ms=0.5):
    """Compare pairwise cross-correlations between neurons.

    Args:
        spike_trains_bio: (n_neurons, T) binary arrays
        spike_trains_surrogate: (n_neurons, T) binary arrays
    """
    from scipy.stats import pearsonr

    bin_size = int(bin_ms / dt_ms)
    n_neurons = spike_trains_bio.shape[0]

    def binned_rates(trains):
        T = trains.shape[1]
        n_bins = T // bin_size
        return trains[:, :n_bins * bin_size].reshape(n_neurons, n_bins, bin_size).sum(axis=2)

    bio_binned = binned_rates(spike_trains_bio)
    surr_binned = binned_rates(spike_trains_surrogate)

    bio_corrs = []
    surr_corrs = []
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            if np.std(bio_binned[i]) > 0 and np.std(bio_binned[j]) > 0:
                r_bio, _ = pearsonr(bio_binned[i], bio_binned[j])
                bio_corrs.append(r_bio)
            if np.std(surr_binned[i]) > 0 and np.std(surr_binned[j]) > 0:
                r_surr, _ = pearsonr(surr_binned[i], surr_binned[j])
                surr_corrs.append(r_surr)

    return {
        'mean_pairwise_cc_bio': np.mean(bio_corrs) if bio_corrs else 0,
        'mean_pairwise_cc_surrogate': np.mean(surr_corrs) if surr_corrs else 0,
        'std_pairwise_cc_bio': np.std(bio_corrs) if bio_corrs else 0,
        'std_pairwise_cc_surrogate': np.std(surr_corrs) if surr_corrs else 0,
    }


def evaluate_fano_stability(spike_counts_bio, spike_counts_surrogate):
    """Compare Fano factors (variance/mean of spike counts).

    Args:
        spike_counts_bio: (n_neurons,) array of spike counts per window
        spike_counts_surrogate: same
    """
    ff_bio = fano_factor(spike_counts_bio)
    ff_surr = fano_factor(spike_counts_surrogate)
    return {
        'fano_bio': ff_bio,
        'fano_surrogate': ff_surr,
        'fano_ratio': ff_surr / ff_bio if ff_bio > 0 else float('inf'),
    }


def run_level2_validation(circuit_bio, circuit_surrogate, dt_ms=0.5):
    """Run all Level 2 metrics.

    Args:
        circuit_bio: dict with 'lfp', 'spike_trains' (n_neurons, T), 'spike_counts'
        circuit_surrogate: same structure

    Returns:
        dict of all Level 2 results
    """
    fs_hz = 1000.0 / dt_ms
    results = {}

    if 'lfp' in circuit_bio and 'lfp' in circuit_surrogate:
        results['oscillations'] = evaluate_oscillation_power(
            circuit_bio['lfp'], circuit_surrogate['lfp'], fs_hz)

    if 'spike_trains' in circuit_bio and 'spike_trains' in circuit_surrogate:
        results['correlations'] = evaluate_pairwise_correlations(
            circuit_bio['spike_trains'], circuit_surrogate['spike_trains'],
            dt_ms=dt_ms)

    if 'spike_counts' in circuit_bio and 'spike_counts' in circuit_surrogate:
        results['fano'] = evaluate_fano_stability(
            circuit_bio['spike_counts'], circuit_surrogate['spike_counts'])

    # Summary
    results['summary'] = {
        'gamma_ratio': results.get('oscillations', {}).get('gamma_ratio', 0),
        'beta_ratio': results.get('oscillations', {}).get('beta_ratio', 0),
        'mean_pairwise_cc': results.get('correlations', {}).get('mean_pairwise_cc_surrogate', 0),
        'fano_ratio': results.get('fano', {}).get('fano_ratio', 0),
    }

    return results
