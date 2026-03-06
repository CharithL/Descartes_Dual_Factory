"""
L5PC DESCARTES -- NEURON Recording Setup

Sets up h.Vector() recorders for the three biological-variable levels:

    Level A  -- Gating variables (m, h, z for each channel mechanism).
                These are the raw kinetic state of every ion channel.

    Level B  -- Effective conductances (gbar * product(gate^exp)) and
                ionic currents (ina, ik, ica, ihcn_Ih).
                Conductances are computed post-hoc from Level A + gbar values.
                Currents are recorded directly from NEURON segments.

    Level C  -- Emergent / aggregate properties (spike times, burst ratio,
                calcium transients, dendritic plateau potentials).
                Computed post-hoc from voltage + [Ca]i traces.

This module records the *raw signals* needed to derive all three levels:
voltage, gating variables, ionic currents, and intracellular calcium.

Channel mechanisms (10 total, matching config.CHANNEL_SPECS):
    NaTa_t, Nap_Et2, K_Pst, K_Tst, SKv3_1, SK_E2, Im, Ih,
    Ca_HVA, Ca_LVAst
"""
import numpy as np

from l5pc.config import (
    CHANNEL_SPECS,
    BAHL_REGIONS,
    NEURON_DT_MS,
    RECORDING_DT_MS,
    DOWNSAMPLE_FACTOR,
    T_STEPS,
)


# ---------------------------------------------------------------------------
# Recording setup
# ---------------------------------------------------------------------------

def setup_recordings(cell):
    """Attach h.Vector() recorders to all segments of interest in the cell.

    Parameters
    ----------
    cell : BahlCell (or any object with a ``get_sections()`` method)
        Must expose ``get_sections()`` returning a dict mapping region
        names (from BAHL_REGIONS) to NEURON Section objects.

    Returns
    -------
    rec : dict
        Nested dictionary structured as::

            rec['time']                        -> h.Vector (time)
            rec[region]['v']                   -> h.Vector (membrane voltage)
            rec[region]['gates'][chan][gate]    -> h.Vector (gating variable)
            rec[region]['currents'][current]   -> h.Vector (ionic current)
            rec[region]['cai']                 -> h.Vector ([Ca]i)

        Where *region* is one of BAHL_REGIONS, *chan* iterates over
        CHANNEL_SPECS, *gate* is 'm'/'h'/'z', and *current* is one of
        'ina', 'ik', 'ica', 'ihcn_Ih'.
    """
    from neuron import h  # noqa: delayed import -- NEURON may not be installed

    sections = cell.get_sections()
    rec = {}

    # Time vector (shared)
    t_vec = h.Vector()
    t_vec.record(h._ref_t)
    rec['time'] = t_vec

    # Ionic currents we want to record directly from segments
    current_names = ['ina', 'ik', 'ica', 'ihcn_Ih']

    for region in BAHL_REGIONS:
        sec = sections.get(region)
        if sec is None:
            continue

        rec[region] = {
            'v': None,
            'gates': {},
            'currents': {},
            'cai': None,
        }

        # Use the centre segment (0.5) for recording
        seg = sec(0.5)

        # --- Membrane voltage ---
        v_vec = h.Vector()
        v_vec.record(seg._ref_v)
        rec[region]['v'] = v_vec

        # --- Gating variables per channel ---
        for chan_name, spec in CHANNEL_SPECS.items():
            # Check if mechanism is inserted in this section
            if not hasattr(seg, chan_name):
                continue

            mech = getattr(seg, chan_name)
            rec[region]['gates'][chan_name] = {}

            for gate in spec['gates']:
                gate_vec = h.Vector()
                try:
                    gate_ref = getattr(mech, f'_ref_{gate}')
                    gate_vec.record(gate_ref)
                    rec[region]['gates'][chan_name][gate] = gate_vec
                except AttributeError:
                    # Gate not present on this segment (mechanism may not
                    # expose it here); skip silently.
                    pass

        # --- Ionic currents ---
        for cur_name in current_names:
            cur_vec = h.Vector()
            ref_attr = f'_ref_{cur_name}'
            try:
                cur_ref = getattr(seg, ref_attr)
                cur_vec.record(cur_ref)
                rec[region]['currents'][cur_name] = cur_vec
            except AttributeError:
                pass

        # --- Intracellular calcium ---
        cai_vec = h.Vector()
        try:
            cai_vec.record(seg._ref_cai)
            rec[region]['cai'] = cai_vec
        except AttributeError:
            pass

    return rec


# ---------------------------------------------------------------------------
# Extraction and downsampling
# ---------------------------------------------------------------------------

def extract_recordings(rec_dict, downsample_factor=None):
    """Convert h.Vector recordings to numpy arrays with downsampling.

    NEURON integrates at NEURON_DT_MS (0.025 ms), but we only need data at
    RECORDING_DT_MS (0.5 ms).  Downsampling by factor 20 (default) selects
    every 20th sample, giving T_STEPS = 2000 time-points per trial.

    Parameters
    ----------
    rec_dict : dict
        Output of ``setup_recordings``.
    downsample_factor : int, optional
        Take every Nth sample.  Defaults to config.DOWNSAMPLE_FACTOR (20).

    Returns
    -------
    data : dict
        Same nested structure as rec_dict but with numpy arrays instead of
        h.Vector objects.  Shape of each array is (T_STEPS,).
    """
    if downsample_factor is None:
        downsample_factor = DOWNSAMPLE_FACTOR

    data = {}

    # Time
    t_raw = np.array(rec_dict['time'])
    data['time'] = t_raw[::downsample_factor]

    for region in BAHL_REGIONS:
        if region not in rec_dict:
            continue

        region_data = {}
        region_rec = rec_dict[region]

        # Voltage
        if region_rec['v'] is not None:
            v_raw = np.array(region_rec['v'])
            region_data['v'] = v_raw[::downsample_factor].astype(np.float32)

        # Gating variables
        region_data['gates'] = {}
        for chan_name, gate_dict in region_rec['gates'].items():
            region_data['gates'][chan_name] = {}
            for gate_name, gate_vec in gate_dict.items():
                g_raw = np.array(gate_vec)
                region_data['gates'][chan_name][gate_name] = (
                    g_raw[::downsample_factor].astype(np.float32)
                )

        # Ionic currents
        region_data['currents'] = {}
        for cur_name, cur_vec in region_rec['currents'].items():
            c_raw = np.array(cur_vec)
            region_data['currents'][cur_name] = (
                c_raw[::downsample_factor].astype(np.float32)
            )

        # Intracellular calcium
        if region_rec['cai'] is not None:
            cai_raw = np.array(region_rec['cai'])
            region_data['cai'] = cai_raw[::downsample_factor].astype(np.float32)

        data[region] = region_data

    return data


# ---------------------------------------------------------------------------
# Variable-name metadata
# ---------------------------------------------------------------------------

def get_variable_names(rec_dict):
    """Build metadata listing all recorded variable names per level.

    Parameters
    ----------
    rec_dict : dict
        Output of ``setup_recordings`` (or ``extract_recordings``).

    Returns
    -------
    names : dict
        'level_A'      : list of str -- gating variable identifiers
                         e.g. 'soma__NaTa_t__m', 'nexus__Ca_HVA__h'
        'level_B_cond' : list of str -- effective conductance identifiers
                         e.g. 'soma__NaTa_t_g', 'nexus__Ca_HVA_g'
        'level_B_curr' : list of str -- ionic current identifiers
                         e.g. 'soma__ina', 'nexus__ica'
        'level_C'      : list of str -- emergent property names
                         (fixed set, independent of recording)
        'voltage'      : list of str -- voltage identifiers per region
        'calcium'      : list of str -- [Ca]i identifiers per region
    """
    level_a = []
    level_b_cond = []
    level_b_curr = []
    voltage_names = []
    calcium_names = []

    for region in BAHL_REGIONS:
        if region not in rec_dict:
            continue

        region_rec = rec_dict[region]

        # Voltage
        if 'v' in region_rec and region_rec['v'] is not None:
            voltage_names.append(f'{region}__v')

        # Level A: gating variables
        gates = region_rec.get('gates', {})
        for chan_name in gates:
            for gate_name in gates[chan_name]:
                level_a.append(f'{region}__{chan_name}__{gate_name}')
            # One effective conductance per channel per region
            level_b_cond.append(f'{region}__{chan_name}_g')

        # Level B currents
        currents = region_rec.get('currents', {})
        for cur_name in currents:
            level_b_curr.append(f'{region}__{cur_name}')

        # Calcium
        cai = region_rec.get('cai')
        if cai is not None:
            calcium_names.append(f'{region}__cai')

    # Level C: emergent properties (fixed list -- computed post-hoc)
    level_c = [
        'spike_count',
        'mean_firing_rate_hz',
        'burst_ratio',
        'first_spike_latency_ms',
        'mean_isi_ms',
        'cv_isi',
        'nexus_calcium_peak',
        'nexus_calcium_mean',
        'tuft_calcium_peak',
        'bac_detected',
        'soma_depolarization_integral',
        'apical_plateau_duration_ms',
    ]

    return {
        'level_A': level_a,
        'level_B_cond': level_b_cond,
        'level_B_curr': level_b_curr,
        'level_C': level_c,
        'voltage': voltage_names,
        'calcium': calcium_names,
    }


# ---------------------------------------------------------------------------
# Flatten recordings into 2-D arrays for storage
# ---------------------------------------------------------------------------

def flatten_level_a(extracted, variable_names):
    """Stack all gating variables into a single (T, N_gates) array.

    Parameters
    ----------
    extracted : dict
        Output of ``extract_recordings``.
    variable_names : dict
        Output of ``get_variable_names``.

    Returns
    -------
    level_a : np.ndarray, shape (T, N_gates), dtype float32
    """
    columns = []
    for var_id in variable_names['level_A']:
        parts = var_id.split('__')
        region, chan, gate = parts[0], parts[1], parts[2]
        col = extracted[region]['gates'][chan][gate]
        columns.append(col)
    if not columns:
        return np.empty((T_STEPS, 0), dtype=np.float32)
    return np.column_stack(columns).astype(np.float32)


def flatten_level_b_currents(extracted, variable_names):
    """Stack all ionic current traces into a single (T, N_curr) array.

    Parameters
    ----------
    extracted : dict
        Output of ``extract_recordings``.
    variable_names : dict
        Output of ``get_variable_names``.

    Returns
    -------
    level_b_curr : np.ndarray, shape (T, N_currents), dtype float32
    """
    columns = []
    for var_id in variable_names['level_B_curr']:
        parts = var_id.split('__')
        region, cur_name = parts[0], parts[1]
        col = extracted[region]['currents'][cur_name]
        columns.append(col)
    if not columns:
        return np.empty((T_STEPS, 0), dtype=np.float32)
    return np.column_stack(columns).astype(np.float32)


def flatten_voltages(extracted, variable_names):
    """Stack all voltage traces into a (T, N_regions) array."""
    columns = []
    for var_id in variable_names['voltage']:
        region = var_id.split('__')[0]
        columns.append(extracted[region]['v'])
    if not columns:
        return np.empty((T_STEPS, 0), dtype=np.float32)
    return np.column_stack(columns).astype(np.float32)
