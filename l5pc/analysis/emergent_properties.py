"""
Level C probing targets: emergent dendritic properties.

These are the L5PC equivalents of the gamma_amp targets used in the
hippocampal circuit — high-level functional observables that depend on
complex nonlinear interactions among multiple ion channels. A surrogate
that merely replays input-output mappings cannot reconstruct these
without genuinely encoding internal biophysical state.
"""
import logging
from pathlib import Path

import numpy as np

from l5pc.config import (
    BURST_ISI_THRESHOLD_MS,
    RECORDING_DT_MS,
    SIM_DURATION_MS,
    T_STEPS,
    BAHL_TRIAL_DIR,
    BAHL_REGIONS,
    CA_HOTZONE_START_UM,
    CA_HOTZONE_END_UM,
    RESULTS_DIR,
    CHANNEL_SPECS,
)
from l5pc.utils.io import load_trial, load_variable_names, save_results_json
from l5pc.utils.metrics import detect_spikes, burst_ratio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core emergent property computation
# ---------------------------------------------------------------------------

def compute_emergent_properties(recordings, spike_times=None, dt_ms=None):
    """Compute higher-order dendritic properties for a single trial.

    These are the Level C probing targets that capture emergent
    functional properties of L5PC dendritic computation:

    - Ca_hotzone_peak:        Peak [Ca2+]i in the nexus/hot zone region (mM)
    - Ca_hotzone_mean:        Mean [Ca2+]i in the hot zone over the trial (mM)
    - burst_ratio:            Fraction of spikes in bursts (ISI < 10ms)
    - dendritic_Ca_amplitude: Peak voltage deflection in tuft during Ca spike (mV)
    - AHP_depth:              Post-spike after-hyperpolarization depth (mV)
    - I_Na_peak:              Maximum sodium current magnitude during AP (negative, nA)
    - mean_firing_rate_hz:    Mean somatic firing rate over the trial (Hz)
    - first_spike_latency_ms: Time to first somatic spike from trial start (ms)
    - mean_isi_ms:            Mean inter-spike interval (ms)
    - cv_isi:                 Coefficient of variation of ISI (dimensionless)
    - bac_detected:           Whether a dendritic Ca spike was detected (0 or 1)
    - n_dendritic_Ca_spikes:  Number of dendritic Ca spike events detected

    Args:
        recordings: dict with keys corresponding to recorded variables.
            Expected keys (when available):
            - 'V_soma':     Somatic voltage timeseries (mV)
            - 'V_tuft':     Tuft voltage timeseries (mV)
            - 'V_nexus':    Nexus voltage timeseries (mV)
            - 'cai_nexus':  Intracellular [Ca2+] at nexus (mM)
            - 'cai_tuft':   Intracellular [Ca2+] at tuft (mM)
            - 'I_NaTa_t_soma': Sodium current at soma (nA)
        spike_times: array of spike times in ms. If None, detected from V_soma.
        dt_ms: timestep in ms. Defaults to config.RECORDING_DT_MS.

    Returns:
        dict of property names to scalar values.
    """
    dt_ms = dt_ms or RECORDING_DT_MS
    props = {}

    # --- Spike detection ---
    V_soma = recordings.get('V_soma', None)
    if spike_times is None and V_soma is not None:
        spike_times = detect_spikes(V_soma, threshold=0.0, dt_ms=dt_ms)

    if spike_times is None:
        spike_times = np.array([])

    # --- Calcium hot zone properties ---
    cai_nexus = recordings.get('cai_nexus', None)
    cai_tuft = recordings.get('cai_tuft', None)

    # Use nexus Ca if available, fall back to tuft
    cai_hotzone = cai_nexus if cai_nexus is not None else cai_tuft

    if cai_hotzone is not None:
        props['Ca_hotzone_peak'] = float(np.max(cai_hotzone))
        props['Ca_hotzone_mean'] = float(np.mean(cai_hotzone))
    else:
        props['Ca_hotzone_peak'] = 0.0
        props['Ca_hotzone_mean'] = 0.0

    # --- Burst ratio ---
    props['burst_ratio'] = burst_ratio(spike_times, burst_isi_ms=BURST_ISI_THRESHOLD_MS)

    # --- Dendritic calcium spike amplitude ---
    V_tuft = recordings.get('V_tuft', None)
    V_nexus = recordings.get('V_nexus', None)
    V_dend = V_tuft if V_tuft is not None else V_nexus

    if V_dend is not None:
        props['dendritic_Ca_amplitude'] = float(np.max(V_dend))
    else:
        props['dendritic_Ca_amplitude'] = 0.0

    # --- AHP depth ---
    if V_soma is not None and len(spike_times) > 0:
        props['AHP_depth'] = _compute_ahp_depth(V_soma, spike_times, dt_ms)
    else:
        props['AHP_depth'] = 0.0

    # --- Peak sodium current ---
    I_Na = recordings.get('I_NaTa_t_soma', None)
    if I_Na is not None:
        # Convention: inward Na current is negative; report magnitude as negative
        props['I_Na_peak'] = float(np.min(I_Na))
    else:
        props['I_Na_peak'] = 0.0

    # --- Spike count (auxiliary) ---
    props['n_spikes'] = int(len(spike_times))

    # --- Firing rate properties ---
    sim_duration_s = (dt_ms * T_STEPS) / 1000.0 if V_soma is not None else SIM_DURATION_MS / 1000.0
    n_spikes = len(spike_times)
    props['mean_firing_rate_hz'] = float(n_spikes / sim_duration_s) if sim_duration_s > 0 else 0.0

    if n_spikes > 0:
        props['first_spike_latency_ms'] = float(spike_times[0])
    else:
        props['first_spike_latency_ms'] = float(SIM_DURATION_MS)  # No spike → max latency

    # --- ISI statistics ---
    if n_spikes >= 2:
        isis = np.diff(spike_times)
        props['mean_isi_ms'] = float(np.mean(isis))
        isi_std = float(np.std(isis))
        props['cv_isi'] = isi_std / props['mean_isi_ms'] if props['mean_isi_ms'] > 0 else 0.0
    else:
        props['mean_isi_ms'] = 0.0
        props['cv_isi'] = 0.0

    # --- BAC detection via dendritic Ca spike events ---
    # A dendritic Ca spike coincident with somatic firing indicates
    # backpropagation-activated calcium firing (BAC).
    V_dend_for_ca = V_tuft if V_tuft is not None else V_nexus
    if V_dend_for_ca is not None:
        ca_events = detect_dendritic_ca_events(V_dend_for_ca, threshold_mv=-20.0,
                                                min_width_ms=5.0, dt_ms=dt_ms)
        props['n_dendritic_Ca_spikes'] = int(len(ca_events))
        props['bac_detected'] = 1.0 if (len(ca_events) > 0 and n_spikes > 0) else 0.0
    else:
        props['n_dendritic_Ca_spikes'] = 0
        props['bac_detected'] = 0.0

    return props


def _compute_ahp_depth(V_soma, spike_times, dt_ms):
    """Compute after-hyperpolarization depth averaged over all spikes.

    AHP depth is the minimum voltage in the 5-20 ms window following
    each spike peak, relative to the resting potential estimated as
    the 10th percentile of the voltage trace.

    Returns:
        Mean AHP depth (mV, negative relative to rest indicates deeper AHP).
    """
    V_rest_estimate = float(np.percentile(V_soma, 10))
    t_indices = (spike_times / dt_ms).astype(int)

    # Post-spike window: 5-20 ms after spike
    window_start = int(5.0 / dt_ms)
    window_end = int(20.0 / dt_ms)

    ahp_depths = []
    for t_idx in t_indices:
        start = t_idx + window_start
        end = t_idx + window_end
        if end >= len(V_soma):
            continue
        V_min = float(np.min(V_soma[start:end]))
        ahp_depths.append(V_min - V_rest_estimate)

    if not ahp_depths:
        return 0.0

    return float(np.mean(ahp_depths))


# ---------------------------------------------------------------------------
# BAC firing index
# ---------------------------------------------------------------------------

def compute_bac_index(combined_spikes, basal_only_spikes, apical_only_spikes):
    """BAC firing index: supralinearity ratio.

    BAC (backpropagation-activated calcium) firing is a hallmark of L5PC
    dendritic computation. When basal and apical inputs coincide, the cell
    fires more than the sum of responses to each input alone.

    BAC_index = n_spikes_combined / (n_spikes_basal_only + n_spikes_apical_only)

    A BAC_index > 1.0 indicates supralinear dendritic integration.
    A BAC_index of ~1.0 indicates linear summation.
    A BAC_index < 1.0 indicates sublinear summation (shunting).

    Args:
        combined_spikes: spike times from trial with both basal + apical input.
        basal_only_spikes: spike times from control trial with basal input only.
        apical_only_spikes: spike times from control trial with apical input only.

    Returns:
        float: BAC firing index. Returns 0.0 if denominator is zero.

    Note:
        Requires separate control trials with basal-only and apical-only input.
        These should use matched stimulation intensities from config.STIM_CONDITIONS.
    """
    n_combined = len(combined_spikes)
    n_basal = len(basal_only_spikes)
    n_apical = len(apical_only_spikes)

    denominator = n_basal + n_apical
    if denominator == 0:
        if n_combined > 0:
            return float('inf')
        return 0.0

    return float(n_combined) / float(denominator)


# ---------------------------------------------------------------------------
# Ca spike detection in dendrite
# ---------------------------------------------------------------------------

def detect_dendritic_ca_events(V_dend, threshold_mv=-20.0, min_width_ms=5.0,
                                dt_ms=None):
    """Detect dendritic calcium spike events from dendritic voltage.

    Ca spikes are broader than Na spikes and have a characteristic plateau.
    We detect them as sustained depolarisations above threshold in the
    dendritic compartment.

    Args:
        V_dend: dendritic voltage timeseries (tuft or nexus).
        threshold_mv: voltage threshold for Ca event detection.
        min_width_ms: minimum duration above threshold to count as Ca spike.
        dt_ms: timestep. Defaults to config.RECORDING_DT_MS.

    Returns:
        List of dicts with 'onset_ms', 'offset_ms', 'peak_mv', 'width_ms'.
    """
    dt_ms = dt_ms or RECORDING_DT_MS
    min_width_samples = int(min_width_ms / dt_ms)

    above = V_dend > threshold_mv
    events = []

    i = 0
    while i < len(above):
        if above[i]:
            onset = i
            while i < len(above) and above[i]:
                i += 1
            offset = i
            width = offset - onset
            if width >= min_width_samples:
                segment = V_dend[onset:offset]
                events.append({
                    'onset_ms': float(onset * dt_ms),
                    'offset_ms': float(offset * dt_ms),
                    'peak_mv': float(np.max(segment)),
                    'width_ms': float(width * dt_ms),
                })
        else:
            i += 1

    return events


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

def patch_level_c_from_saved_data(trial_dir=None):
    """Recompute Level C properties from saved somatic voltage in existing trial files.

    This is a lightweight alternative to re-simulation that adds spike-timing
    properties (mean_firing_rate_hz, first_spike_latency_ms, mean_isi_ms,
    cv_isi) and an approximate bac_detected flag to existing trial data.

    The bac_detected flag is derived from the already-stored
    dendritic_Ca_amplitude: if the peak dendritic voltage exceeded -20 mV
    (the Ca-spike threshold used in detect_dendritic_ca_events) AND the
    trial had somatic spikes, we mark bac_detected = 1.

    This function modifies trial .npz files IN-PLACE and updates
    variable_names.json.

    Args:
        trial_dir: directory containing trial .npz files.
            Defaults to config.BAHL_TRIAL_DIR.

    Returns:
        int: number of trials patched.
    """
    trial_dir = Path(trial_dir or BAHL_TRIAL_DIR)

    var_meta = load_variable_names(trial_dir)
    old_keys = None
    if var_meta:
        for mk in ['level_C_keys', 'level_C']:
            candidate = var_meta.get(mk)
            if isinstance(candidate, list) and len(candidate) > 0:
                old_keys = candidate
                break

    trial_files = sorted(trial_dir.glob('trial_*.npz'))
    if not trial_files:
        logger.warning("No trial files in %s, nothing to patch.", trial_dir)
        return 0

    new_keys = None
    n_patched = 0

    for fpath in trial_files:
        data = dict(np.load(fpath))

        # Somatic voltage
        V_soma = data.get('output', None)
        if V_soma is None:
            continue
        V_soma = V_soma.flatten()

        dt_ms = RECORDING_DT_MS
        spike_times = detect_spikes(V_soma, threshold=0.0, dt_ms=dt_ms)
        n_spikes = len(spike_times)
        sim_duration_s = SIM_DURATION_MS / 1000.0

        # Build new properties dict from somatic voltage
        new_props = {}
        new_props['mean_firing_rate_hz'] = float(n_spikes / sim_duration_s) if sim_duration_s > 0 else 0.0
        new_props['first_spike_latency_ms'] = float(spike_times[0]) if n_spikes > 0 else float(SIM_DURATION_MS)

        if n_spikes >= 2:
            isis = np.diff(spike_times)
            new_props['mean_isi_ms'] = float(np.mean(isis))
            isi_std = float(np.std(isis))
            new_props['cv_isi'] = isi_std / new_props['mean_isi_ms'] if new_props['mean_isi_ms'] > 0 else 0.0
        else:
            new_props['mean_isi_ms'] = 0.0
            new_props['cv_isi'] = 0.0

        # Approximate BAC from existing dendritic_Ca_amplitude
        # The existing Level C stores dendritic_Ca_amplitude = max(V_dend).
        # If V_dend ever exceeded -20 mV (Ca spike threshold) AND there were
        # somatic spikes, we flag BAC detected.
        existing_level_c = data.get('level_C_emerge', None)
        dend_ca_amp = None
        if existing_level_c is not None and old_keys is not None:
            if 'dendritic_Ca_amplitude' in old_keys:
                dend_ca_amp_idx = old_keys.index('dendritic_Ca_amplitude')
                if dend_ca_amp_idx < len(existing_level_c):
                    dend_ca_amp = float(existing_level_c[dend_ca_amp_idx])

        if dend_ca_amp is not None:
            new_props['bac_detected'] = 1.0 if (dend_ca_amp > -20.0 and n_spikes > 0) else 0.0
        else:
            new_props['bac_detected'] = 0.0
        # n_dendritic_Ca_spikes can only be computed from raw V_dend
        # traces, which aren't saved. Set to -1 to indicate "unavailable"
        # (will be computed correctly on next re-simulation).
        new_props['n_dendritic_Ca_spikes'] = -1

        # Merge: keep all existing properties, add/overwrite new ones
        merged = {}
        if existing_level_c is not None and old_keys is not None:
            for j, k in enumerate(old_keys):
                if j < len(existing_level_c):
                    merged[k] = float(existing_level_c[j])
        merged.update(new_props)

        # Convert to sorted array
        merged_keys = sorted(merged.keys())
        level_c_array = np.array(
            [merged[k] for k in merged_keys], dtype=np.float32
        )

        if new_keys is None:
            new_keys = merged_keys

        # Re-save with updated Level C
        data['level_C_emerge'] = level_c_array
        np.savez_compressed(fpath, **data)
        n_patched += 1

    # Update variable_names.json
    if new_keys and var_meta is not None:
        import json
        var_meta['level_C_keys'] = new_keys
        meta_path = trial_dir / 'variable_names.json'
        with open(meta_path, 'w') as f:
            json.dump(var_meta, f, indent=2)
        logger.info("Updated level_C_keys in %s: %s", meta_path, new_keys)

    logger.info("Patched Level C in %d trials (now %d properties).",
                n_patched, len(new_keys or []))
    return n_patched


def compute_all_emergent(trial_dir=None, save_path=None, n_trials=None):
    """Compute Level C targets for all trials and save.

    Args:
        trial_dir: directory containing trial .npz files.
            Defaults to config.BAHL_TRIAL_DIR.
        save_path: directory to save Level C results.
            Defaults to config.RESULTS_DIR / 'level_C'.
        n_trials: number of trials to process. Defaults to all found.

    Returns:
        Summary dict with property names and per-trial values.
    """
    trial_dir = Path(trial_dir or BAHL_TRIAL_DIR)
    save_path = Path(save_path or RESULTS_DIR / 'level_C')
    save_path.mkdir(parents=True, exist_ok=True)

    var_names = load_variable_names(trial_dir)

    # Discover trials
    trial_files = sorted(trial_dir.glob('trial_*.npz'))
    if n_trials is not None:
        trial_files = trial_files[:n_trials]

    if not trial_files:
        logger.error("No trial files found in %s", trial_dir)
        return {}

    logger.info("Computing Level C properties for %d trials.", len(trial_files))

    all_properties = []
    property_names = None

    for fpath in trial_files:
        idx = int(fpath.stem.split('_')[1])
        trial_data = load_trial(trial_dir, idx)

        recordings = _build_recordings_dict(trial_data, var_names)
        props = compute_emergent_properties(recordings)
        all_properties.append(props)

        if property_names is None:
            property_names = sorted(props.keys())

    # Aggregate into arrays
    aggregated = {}
    for name in (property_names or []):
        aggregated[name] = [p.get(name, 0.0) for p in all_properties]

    # Save per-trial values as numpy array
    if property_names:
        prop_matrix = np.array(
            [[p.get(name, 0.0) for name in property_names] for p in all_properties]
        )
        np.savez_compressed(
            save_path / 'level_C_properties.npz',
            properties=prop_matrix,
            property_names=np.array(property_names),
        )

    # Save summary statistics
    summary = {
        'property_names': property_names or [],
        'n_trials': len(trial_files),
        'means': {name: float(np.mean(vals)) for name, vals in aggregated.items()},
        'stds': {name: float(np.std(vals)) for name, vals in aggregated.items()},
        'source_dir': str(trial_dir),
    }
    save_results_json(summary, save_path / 'level_C_summary.json')

    logger.info(
        "Level C complete: %d properties across %d trials.",
        len(property_names or []),
        len(trial_files),
    )
    return summary


def _build_recordings_dict(trial_data, var_names):
    """Build a recordings dict suitable for compute_emergent_properties.

    Maps trial .npz fields to named recording channels based on
    variable name metadata.
    """
    recordings = {}

    # Somatic voltage from output
    if 'output' in trial_data:
        recordings['V_soma'] = trial_data['output'].flatten()

    # Level C pre-computed emergent data (if stored during simulation)
    if 'level_C_emerge' in trial_data:
        emerge = trial_data['level_C_emerge']
        if var_names and 'level_C' in var_names:
            for j, name in enumerate(var_names['level_C']):
                if j < emerge.shape[-1]:
                    recordings[name] = emerge[..., j]

    # Multi-region voltages
    if 'voltages' in trial_data and var_names and 'voltages' in var_names:
        v_data = trial_data['voltages']
        for j, region in enumerate(var_names['voltages']):
            if j < v_data.shape[-1]:
                recordings[f'V_{region}'] = v_data[..., j]

    # Calcium concentrations
    if 'calcium' in trial_data and var_names and 'calcium' in var_names:
        ca_data = trial_data['calcium']
        for j, name in enumerate(var_names['calcium']):
            if j < ca_data.shape[-1]:
                recordings[name] = ca_data[..., j]

    # Ionic currents (if precomputed at Level B)
    if 'level_B_curr' in trial_data and var_names and 'level_B_currents' in var_names:
        curr_data = trial_data['level_B_curr']
        for j, name in enumerate(var_names['level_B_currents']):
            if j < curr_data.shape[-1]:
                recordings[name] = curr_data[..., j]

    return recordings
