"""
L5PC DESCARTES -- Synaptic Input Pattern Generation

Generates Poisson-distributed spike trains for 5 stimulation conditions.
Basal and apical inputs are SEPARATE channels to enable coincidence detection
analysis (BAC firing requires near-simultaneous basal + apical input).

Conditions:
    subthreshold  Low-rate basal only; no/minimal apical. No spikes expected.
    tonic         Moderate basal drive; cell fires regularly.
    burst         High basal + apical; high-frequency bursting.
    bac           Moderate basal + strong apical; back-propagating AP
                  activated calcium spike (BAC firing).
    mixed         Full-range random rates; diverse regime coverage.

All parameters are drawn from config.py -- nothing is hardcoded here.
"""
import numpy as np

from l5pc.config import (
    STIM_CONDITIONS,
    N_BASAL_SYN,
    N_APICAL_SYN,
    N_SOMA_SYN,
    SIM_DURATION_MS,
    NEURON_DT_MS,
)


# ---------------------------------------------------------------------------
# Core spike generation
# ---------------------------------------------------------------------------

def generate_poisson_spikes(rate_hz, duration_ms, n_synapses, dt_ms=None):
    """Generate independent Poisson spike trains for a population of synapses.

    Parameters
    ----------
    rate_hz : float
        Mean firing rate in Hz for each synapse.
    duration_ms : float
        Duration of the simulation window in milliseconds.
    n_synapses : int
        Number of independent synapses (columns of output).
    dt_ms : float, optional
        Time-step resolution in milliseconds.  Defaults to NEURON_DT_MS
        from config (0.025 ms) so that the spike trains live on the same
        time grid as the NEURON integrator.

    Returns
    -------
    spikes : np.ndarray, shape (T, n_synapses), dtype np.float32
        Binary array where 1 indicates a spike in that time bin.
        T = int(duration_ms / dt_ms).
    """
    if dt_ms is None:
        dt_ms = NEURON_DT_MS

    n_steps = int(duration_ms / dt_ms)

    if rate_hz <= 0.0 or n_synapses == 0:
        return np.zeros((n_steps, n_synapses), dtype=np.float32)

    # Probability of a spike in one dt bin
    p_spike = rate_hz * (dt_ms / 1000.0)
    # Clamp to valid probability range
    p_spike = min(p_spike, 1.0)

    spikes = (np.random.rand(n_steps, n_synapses) < p_spike).astype(np.float32)
    return spikes


# ---------------------------------------------------------------------------
# Per-condition input generation
# ---------------------------------------------------------------------------

def _sample_rate(rate_range):
    """Draw a uniform random rate from (low, high) Hz tuple."""
    low, high = rate_range
    if low >= high:
        return float(low)
    return np.random.uniform(low, high)


def generate_trial_inputs(condition_name, condition_params, duration_ms=None):
    """Generate a full set of synaptic inputs for one trial under a condition.

    Basal (excitatory), apical (excitatory), and somatic (inhibitory) inputs
    are generated as **independent** Poisson channels.  This separation is
    critical: the coincidence-detection analysis in DESCARTES depends on
    being able to manipulate basal vs. apical timing independently.

    Parameters
    ----------
    condition_name : str
        Label for the stimulation condition (e.g. 'bac', 'tonic').
    condition_params : dict
        Must contain 'basal_hz' and 'apical_hz' as (low, high) tuples.
    duration_ms : float, optional
        Override for simulation duration (defaults to config.SIM_DURATION_MS).

    Returns
    -------
    trial : dict
        'basal'     : np.ndarray (T, N_BASAL_SYN)   -- excitatory
        'apical'    : np.ndarray (T, N_APICAL_SYN)   -- excitatory
        'soma'      : np.ndarray (T, N_SOMA_SYN)     -- inhibitory (GABA)
        'condition'  : str                            -- condition label
        'basal_rate' : float                          -- sampled rate (Hz)
        'apical_rate': float                          -- sampled rate (Hz)
        'soma_rate'  : float                          -- sampled rate (Hz)
    """
    if duration_ms is None:
        duration_ms = SIM_DURATION_MS

    basal_rate = _sample_rate(condition_params['basal_hz'])
    apical_rate = _sample_rate(condition_params['apical_hz'])

    # Somatic inhibition scales with total excitation to maintain balance.
    # Inhibitory rate is proportional to the geometric mean of excitatory
    # rates, with a modest floor so there is always some tonic inhibition.
    soma_rate = max(1.0, 0.5 * (basal_rate + apical_rate))

    basal_spikes = generate_poisson_spikes(basal_rate, duration_ms, N_BASAL_SYN)
    apical_spikes = generate_poisson_spikes(apical_rate, duration_ms, N_APICAL_SYN)
    soma_spikes = generate_poisson_spikes(soma_rate, duration_ms, N_SOMA_SYN)

    return {
        'basal': basal_spikes,
        'apical': apical_spikes,
        'soma': soma_spikes,
        'condition': condition_name,
        'basal_rate': float(basal_rate),
        'apical_rate': float(apical_rate),
        'soma_rate': float(soma_rate),
    }


# ---------------------------------------------------------------------------
# Batch generation across all conditions
# ---------------------------------------------------------------------------

def generate_all_trials(stim_conditions=None, duration_ms=None, seed=None):
    """Generate inputs for all trials across every stimulation condition.

    Parameters
    ----------
    stim_conditions : dict, optional
        Mapping of condition_name -> params dict.  Each params dict must
        contain 'basal_hz', 'apical_hz' (rate range tuples) and 'n_trials'.
        Defaults to config.STIM_CONDITIONS.
    duration_ms : float, optional
        Override simulation duration (defaults to config.SIM_DURATION_MS).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    all_trials : list of dict
        500 trial-input dicts (by default), globally shuffled so that
        conditions are interleaved.  Each dict has the same structure as
        the output of ``generate_trial_inputs``, plus a 'trial_idx' field
        assigned after shuffling.
    """
    if stim_conditions is None:
        stim_conditions = STIM_CONDITIONS
    if duration_ms is None:
        duration_ms = SIM_DURATION_MS
    if seed is not None:
        np.random.seed(seed)

    all_trials = []

    for cond_name, cond_params in stim_conditions.items():
        n_trials = cond_params['n_trials']
        for _ in range(n_trials):
            trial = generate_trial_inputs(cond_name, cond_params, duration_ms)
            all_trials.append(trial)

    # Shuffle so conditions are interleaved (important for train/val/test
    # splits to contain all conditions).
    np.random.shuffle(all_trials)

    # Assign sequential trial indices after shuffling
    for idx, trial in enumerate(all_trials):
        trial['trial_idx'] = idx

    return all_trials
