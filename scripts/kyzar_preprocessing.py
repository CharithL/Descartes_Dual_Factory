"""
kyzar/preprocessing.py

DESCARTES Circuit 6 Phase 1.1: Spike train extraction, binning,
trial segmentation, and I/O tensor creation for Kyzar Sternberg data.

Processes NWB files from DANDI 000469. Extracts continuous spike
timestamps, bins at 10ms, Gaussian smooths at 30ms, segments into
trials aligned to fixation onset, and creates LSTM-ready tensors.
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

log = logging.getLogger(__name__)

# Region classification for DESCARTES I/O
LIMBIC_REGIONS = {'hippocampus', 'amygdala'}
FRONTAL_REGIONS = {'dACC', 'preSMA', 'vmPFC'}

# Epoch codes for the epoch mask
EPOCH_FIXATION = 0
EPOCH_ENCODING = 1
EPOCH_MAINTENANCE = 2
EPOCH_PROBE = 3
EPOCH_RESPONSE = 4


def normalize_region(location_str):
    """Map NWB electrode location string to standardized region."""
    loc = location_str.lower().replace(' ', '_')
    if 'hippocampus' in loc:
        return 'hippocampus'
    elif 'amygdala' in loc:
        return 'amygdala'
    elif 'anterior_cingulate' in loc or 'dacc' in loc:
        return 'dACC'
    elif 'supplementary_motor' in loc or 'pre_sma' in loc:
        return 'preSMA'
    elif 'prefrontal' in loc or 'vmpfc' in loc:
        return 'vmPFC'
    return loc


def extract_units(nwb, min_fr=0.5):
    """Extract spike times per unit with region labels.

    Args:
        nwb: opened NWB file object
        min_fr: minimum firing rate threshold (Hz)

    Returns:
        list of dicts: [{unit_id, region, spike_times, firing_rate}, ...]
    """
    units = []
    if nwb.units is None or len(nwb.units) == 0:
        return units

    for i in range(len(nwb.units)):
        spikes = nwb.units['spike_times'][i]
        n_spikes = len(spikes)

        electrode_idx = nwb.units['electrodes'][i].index[0]
        raw_location = nwb.electrodes['location'][electrode_idx]
        region = normalize_region(raw_location)

        if n_spikes >= 2:
            duration = spikes[-1] - spikes[0]
            fr = n_spikes / duration if duration > 0 else 0
        else:
            fr = 0

        if fr < min_fr:
            continue

        units.append({
            'unit_id': i,
            'region': region,
            'spike_times': np.array(spikes, dtype=np.float64),
            'firing_rate': float(fr),
            'is_limbic': region in LIMBIC_REGIONS,
            'is_frontal': region in FRONTAL_REGIONS,
        })

    return units


def bin_spikes(spike_times, bin_edges):
    """Bin spike times into time bins.

    Args:
        spike_times: array of spike timestamps (seconds)
        bin_edges: array of bin edges (seconds), length T+1

    Returns:
        counts: array of spike counts per bin, length T
    """
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    return counts.astype(np.float64)


def smooth_rates(binned, sigma_bins):
    """Gaussian smooth binned spike counts.

    Args:
        binned: (T,) array of spike counts
        sigma_bins: sigma in units of bins (e.g., 3 for 30ms at 10ms bins)

    Returns:
        smoothed: (T,) array of smoothed firing rates
    """
    if sigma_bins <= 0:
        return binned
    return gaussian_filter1d(binned, sigma=sigma_bins, mode='nearest')


def extract_trial_epochs(nwb, trial_idx):
    """Extract epoch timestamps for one trial.

    Returns dict with epoch onset times (seconds), or None values
    for absent epochs (e.g., enc2/enc3 in load-1 trials).
    """
    trials = nwb.trials

    epoch = {
        'fixation': trials['timestamps_FixationCross'][trial_idx],
        'encoding1': trials['timestamps_Encoding1'][trial_idx],
        'encoding1_end': trials['timestamps_Encoding1_end'][trial_idx],
        'encoding2': trials['timestamps_Encoding2'][trial_idx],
        'encoding2_end': trials['timestamps_Encoding2_end'][trial_idx],
        'encoding3': trials['timestamps_Encoding3'][trial_idx],
        'encoding3_end': trials['timestamps_Encoding3_end'][trial_idx],
        'maintenance': trials['timestamps_Maintenance'][trial_idx],
        'probe': trials['timestamps_Probe'][trial_idx],
        'response': trials['timestamps_Response'][trial_idx],
        'start': trials['start_time'][trial_idx],
        'stop': trials['stop_time'][trial_idx],
    }

    # Load and behavioral data
    epoch['load'] = int(trials['loads'][trial_idx])
    epoch['accuracy'] = int(trials['response_accuracy'][trial_idx])
    epoch['probe_in_out'] = int(trials['probe_in_out'][trial_idx])

    # Image IDs
    epoch['enc1_pic'] = int(trials['loadsEnc1_PicIDs'][trial_idx])
    epoch['probe_pic'] = int(trials['loadsProbe_PicIDs'][trial_idx])

    return epoch


def create_epoch_mask(epoch_info, bin_edges):
    """Create epoch identity mask for one trial.

    Args:
        epoch_info: dict from extract_trial_epochs
        bin_edges: array of bin edges (seconds)

    Returns:
        mask: (T,) array with epoch codes (0-4)
    """
    T = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = np.zeros(T, dtype=np.int32)

    fix_t = epoch_info['fixation']
    enc1_t = epoch_info['encoding1']
    maint_t = epoch_info['maintenance']
    probe_t = epoch_info['probe']
    resp_t = epoch_info['response']

    for i, t in enumerate(bin_centers):
        if t < enc1_t:
            mask[i] = EPOCH_FIXATION
        elif t < maint_t:
            mask[i] = EPOCH_ENCODING
        elif t < probe_t:
            mask[i] = EPOCH_MAINTENANCE
        elif t < resp_t:
            mask[i] = EPOCH_PROBE
        else:
            mask[i] = EPOCH_RESPONSE

    return mask


def process_session(nwb_path, output_dir, dt_ms=10, sigma_ms=30,
                    min_fr=0.5, pre_ms=0, post_ms=500):
    """Process one Sternberg NWB session into LSTM-ready tensors.

    Args:
        nwb_path: path to NWB file
        output_dir: where to save processed data
        dt_ms: bin width in milliseconds
        sigma_ms: Gaussian smoothing sigma in milliseconds
        min_fr: minimum firing rate threshold (Hz)
        pre_ms: time before fixation to include (ms)
        post_ms: time after response to include (ms)

    Returns:
        dict with processing summary
    """
    import pynwb

    nwb_path = Path(nwb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    io = pynwb.NWBHDF5IO(str(nwb_path), 'r')
    nwb = io.read()

    subject_id = str(nwb.subject.subject_id) if nwb.subject else 'unknown'
    session_id = str(nwb.session_id)

    # Check this is a Sternberg session
    if 'loads' not in (nwb.trials.colnames if nwb.trials else []):
        io.close()
        raise ValueError(f"{nwb_path.name} is not a Sternberg session")

    log.info("Processing sub-%s ses-%s", subject_id, session_id)

    # Extract units
    units = extract_units(nwb, min_fr=min_fr)
    limbic_units = [u for u in units if u['is_limbic']]
    frontal_units = [u for u in units if u['is_frontal']]

    n_limbic = len(limbic_units)
    n_frontal = len(frontal_units)
    log.info("  Units: %d limbic, %d frontal (%d total)",
             n_limbic, n_frontal, len(units))

    if n_limbic < 5 or n_frontal < 5:
        io.close()
        raise ValueError(
            f"Insufficient neurons: {n_limbic} limbic, {n_frontal} frontal")

    # Bin parameters
    dt_s = dt_ms / 1000.0
    sigma_bins = sigma_ms / dt_ms  # e.g., 30ms / 10ms = 3 bins
    pre_s = pre_ms / 1000.0
    post_s = post_ms / 1000.0

    # Process each trial
    n_trials = len(nwb.trials)
    X_trials = []
    Y_trials = []
    epoch_masks = []
    trial_metadata = []

    for ti in range(n_trials):
        epoch_info = extract_trial_epochs(nwb, ti)

        # Trial window: fixation - pre_s to response + post_s
        t_start = epoch_info['fixation'] - pre_s
        t_end = epoch_info['response'] + post_s

        # Create bin edges
        bin_edges = np.arange(t_start, t_end + dt_s, dt_s)
        T = len(bin_edges) - 1

        if T < 10:  # Skip very short trials
            continue

        # Bin and smooth each limbic unit
        x_trial = np.zeros((T, n_limbic))
        for ui, unit in enumerate(limbic_units):
            counts = bin_spikes(unit['spike_times'], bin_edges)
            x_trial[:, ui] = smooth_rates(counts, sigma_bins)

        # Bin and smooth each frontal unit
        y_trial = np.zeros((T, n_frontal))
        for ui, unit in enumerate(frontal_units):
            counts = bin_spikes(unit['spike_times'], bin_edges)
            y_trial[:, ui] = smooth_rates(counts, sigma_bins)

        # Epoch mask
        emask = create_epoch_mask(epoch_info, bin_edges)

        # Trial metadata
        rt_ms = (epoch_info['response'] - epoch_info['probe']) * 1000
        meta = {
            'trial_idx': ti,
            'load': epoch_info['load'],
            'accuracy': epoch_info['accuracy'],
            'probe_in_out': epoch_info['probe_in_out'],
            'reaction_time_ms': float(rt_ms),
            'n_bins': T,
            'enc1_pic': epoch_info['enc1_pic'],
            'probe_pic': epoch_info['probe_pic'],
        }

        X_trials.append(x_trial)
        Y_trials.append(y_trial)
        epoch_masks.append(emask)
        trial_metadata.append(meta)

    io.close()

    n_processed = len(X_trials)
    log.info("  Processed %d/%d trials", n_processed, n_trials)

    # Save as variable-length arrays (trials have different durations)
    np.savez(output_dir / 'X_trials.npz',
             **{f'trial_{i}': x for i, x in enumerate(X_trials)})
    np.savez(output_dir / 'Y_trials.npz',
             **{f'trial_{i}': y for i, y in enumerate(Y_trials)})
    np.savez(output_dir / 'epoch_masks.npz',
             **{f'trial_{i}': m for i, m in enumerate(epoch_masks)})

    # Save metadata
    metadata = {
        'subject_id': subject_id,
        'session_id': session_id,
        'nwb_path': str(nwb_path),
        'n_trials': n_processed,
        'n_limbic': n_limbic,
        'n_frontal': n_frontal,
        'dt_ms': dt_ms,
        'sigma_ms': sigma_ms,
        'min_fr': min_fr,
        'limbic_units': [{
            'unit_id': u['unit_id'],
            'region': u['region'],
            'firing_rate': u['firing_rate'],
        } for u in limbic_units],
        'frontal_units': [{
            'unit_id': u['unit_id'],
            'region': u['region'],
            'firing_rate': u['firing_rate'],
        } for u in frontal_units],
        'trial_metadata': trial_metadata,
        'trial_lengths': [x.shape[0] for x in X_trials],
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    log.info("  Saved to %s", output_dir)

    return metadata


def process_all_sessions(data_dir, output_base, passing_subjects=None,
                         dt_ms=10, sigma_ms=30, min_fr=0.5):
    """Process all passing Sternberg sessions.

    Args:
        data_dir: path to kyzar_raw/ with NWB files
        output_base: base path for processed output
        passing_subjects: list of subject IDs to process (default: all)
    """
    data_dir = Path(data_dir)
    output_base = Path(output_base)

    # Find all ses-2 (Sternberg) files
    nwb_files = sorted(data_dir.glob('**/sub-*_ses-2_*.nwb'))

    results = []
    for nwb_path in nwb_files:
        # Extract subject ID from filename
        fname = nwb_path.stem
        sub_id = fname.split('_')[0].replace('sub-', '')

        if passing_subjects and sub_id not in passing_subjects:
            log.info("Skipping sub-%s (not in passing list)", sub_id)
            continue

        out_dir = output_base / f'session_sub{sub_id}_ses2'

        try:
            meta = process_session(
                nwb_path, out_dir,
                dt_ms=dt_ms, sigma_ms=sigma_ms, min_fr=min_fr)
            results.append(meta)
        except Exception as exc:
            log.warning("FAILED sub-%s: %s", sub_id, exc)
            results.append({
                'subject_id': sub_id, 'error': str(exc)})

    # Summary
    n_ok = sum(1 for r in results if 'error' not in r)
    log.info("Processed %d/%d sessions successfully", n_ok, len(results))

    return results
