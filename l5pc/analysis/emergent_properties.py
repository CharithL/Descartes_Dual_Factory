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

    - Ca_hotzone_peak:     Peak [Ca2+]i in the nexus/hot zone region (mM)
    - Ca_hotzone_mean:     Mean [Ca2+]i in the hot zone over the trial (mM)
    - burst_ratio:         Fraction of spikes in bursts (ISI < 10ms)
    - dendritic_Ca_amplitude: Peak voltage deflection in tuft during Ca spike (mV)
    - AHP_depth:           Post-spike after-hyperpolarization depth (mV)
    - I_Na_peak:           Maximum sodium current magnitude during AP (negative, nA)

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
