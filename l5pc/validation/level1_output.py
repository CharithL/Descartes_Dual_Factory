"""Level 1 Validation: Output Fidelity.

Measures whether the surrogate produces the same spike output as the
biological neuron. This is the minimum requirement for functional equivalence.
"""
import numpy as np
from l5pc.utils.metrics import (
    cross_condition_correlation, detect_spikes, ks_test_isi,
    victor_purpura_distance, burst_ratio
)


def evaluate_spike_rate(bio_output, surrogate_output, dt_ms=0.5):
    """Compare mean firing rates.

    Returns:
        dict with bio_rate, surrogate_rate, ratio
    """
    bio_spikes = detect_spikes(bio_output, dt_ms=dt_ms)
    surr_spikes = detect_spikes(surrogate_output, dt_ms=dt_ms)
    duration_s = len(bio_output) * dt_ms / 1000.0

    bio_rate = len(bio_spikes) / duration_s if duration_s > 0 else 0
    surr_rate = len(surr_spikes) / duration_s if duration_s > 0 else 0
    ratio = surr_rate / bio_rate if bio_rate > 0 else float('inf')

    return {'bio_rate_hz': bio_rate, 'surrogate_rate_hz': surr_rate, 'ratio': ratio}


def evaluate_isi_match(bio_output, surrogate_output, dt_ms=0.5):
    """KS test on ISI distributions."""
    bio_spikes = detect_spikes(bio_output, dt_ms=dt_ms)
    surr_spikes = detect_spikes(surrogate_output, dt_ms=dt_ms)
    stat, p = ks_test_isi(bio_spikes, surr_spikes)
    return {'ks_statistic': stat, 'p_value': p}


def evaluate_vp_distance(bio_output, surrogate_output, dt_ms=0.5, q=1.0):
    """Victor-Purpura spike distance."""
    bio_spikes = detect_spikes(bio_output, dt_ms=dt_ms)
    surr_spikes = detect_spikes(surrogate_output, dt_ms=dt_ms)
    dist = victor_purpura_distance(bio_spikes, surr_spikes, q=q)
    return {'vp_distance': dist, 'q': q}


def evaluate_cross_condition_cc(bio_rates_per_condition, surr_rates_per_condition):
    """Cross-condition correlation — THE metric that works."""
    cc = cross_condition_correlation(
        np.array(bio_rates_per_condition),
        np.array(surr_rates_per_condition)
    )
    return {'cross_condition_cc': cc}


def evaluate_burst_ratio_match(bio_output, surrogate_output, dt_ms=0.5):
    """Compare burst/tonic ratios."""
    bio_spikes = detect_spikes(bio_output, dt_ms=dt_ms)
    surr_spikes = detect_spikes(surrogate_output, dt_ms=dt_ms)
    bio_br = burst_ratio(bio_spikes)
    surr_br = burst_ratio(surr_spikes)
    return {'bio_burst_ratio': bio_br, 'surrogate_burst_ratio': surr_br,
            'difference': abs(bio_br - surr_br)}


def run_level1_validation(bio_outputs, surrogate_outputs, condition_labels, dt_ms=0.5):
    """Run all Level 1 metrics across trials.

    Args:
        bio_outputs: list of (T,) arrays — biological spike trains
        surrogate_outputs: list of (T,) arrays — surrogate spike trains
        condition_labels: list of condition names per trial

    Returns:
        dict of metric results
    """
    results = {
        'spike_rates': [],
        'isi_matches': [],
        'vp_distances': [],
        'burst_ratios': [],
    }

    for bio, surr in zip(bio_outputs, surrogate_outputs):
        results['spike_rates'].append(evaluate_spike_rate(bio, surr, dt_ms))
        results['isi_matches'].append(evaluate_isi_match(bio, surr, dt_ms))
        results['vp_distances'].append(evaluate_vp_distance(bio, surr, dt_ms))
        results['burst_ratios'].append(evaluate_burst_ratio_match(bio, surr, dt_ms))

    # Cross-condition CC (aggregate by condition)
    unique_conditions = sorted(set(condition_labels))
    bio_rates = []
    surr_rates = []
    for cond in unique_conditions:
        cond_idx = [i for i, c in enumerate(condition_labels) if c == cond]
        bio_rates.append(np.mean([results['spike_rates'][i]['bio_rate_hz'] for i in cond_idx]))
        surr_rates.append(np.mean([results['spike_rates'][i]['surrogate_rate_hz'] for i in cond_idx]))
    results['cross_condition'] = evaluate_cross_condition_cc(bio_rates, surr_rates)

    # Summary
    results['summary'] = {
        'mean_rate_ratio': np.mean([r['ratio'] for r in results['spike_rates']]),
        'mean_vp_distance': np.mean([r['vp_distance'] for r in results['vp_distances']]),
        'mean_ks_p': np.mean([r['p_value'] for r in results['isi_matches']]),
        'cross_condition_cc': results['cross_condition']['cross_condition_cc'],
    }

    return results
