"""
Level B probing targets: effective conductances and ionic currents.

Computes identifiable combinations G_eff = gbar * m^p * h^q for each
ion channel, and the resulting ionic currents I = G_eff * (V - E_rev).
These are the "internal state" variables that surrogates should encode
if they truly learn biophysical dynamics rather than input-output shortcuts.
"""
import logging
import re
from pathlib import Path

import numpy as np

from l5pc.config import (
    CHANNEL_SPECS,
    REVERSAL_POTENTIALS,
    BAHL_REGIONS,
    RECORDING_DT_MS,
    T_STEPS,
    BAHL_TRIAL_DIR,
    RESULTS_DIR,
)
from l5pc.utils.io import load_trial, load_variable_names, save_results_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate key parsing
# ---------------------------------------------------------------------------

_GATE_KEY_RE = re.compile(
    r'^(?P<gate>[a-zA-Z])_(?P<channel>[a-zA-Z0-9_]+)_(?P<region>[a-zA-Z_]+)$'
)


def _parse_gate_key(key):
    """Parse a gate variable key like 'm_NaTa_t_soma' into components.

    Returns:
        (gate_letter, channel_name, region) or None if not parseable.
    """
    # Channels can contain underscores (e.g. NaTa_t, Ca_HVA), so we match
    # the gate letter first, then try each known channel name.
    parts = key.split('_', 1)
    if len(parts) < 2:
        return None
    gate_letter = parts[0]
    if len(gate_letter) != 1:
        return None
    remainder = parts[1]  # e.g. 'NaTa_t_soma'

    for channel in CHANNEL_SPECS:
        if remainder.startswith(channel):
            suffix = remainder[len(channel):]
            if suffix.startswith('_') and len(suffix) > 1:
                region = suffix[1:]
                return gate_letter, channel, region
    return None


def _group_gates(gates_dict):
    """Group gate timeseries by (channel, region).

    Args:
        gates_dict: flat dict mapping 'gate_channel_region' to arrays.

    Returns:
        Nested dict: {channel: {region: {gate_letter: array}}}
    """
    grouped = {}
    for key, arr in gates_dict.items():
        parsed = _parse_gate_key(key)
        if parsed is None:
            continue
        gate_letter, channel, region = parsed
        grouped.setdefault(channel, {}).setdefault(region, {})[gate_letter] = arr
    return grouped


# ---------------------------------------------------------------------------
# Effective conductance computation
# ---------------------------------------------------------------------------

def compute_effective_conductances(gates_dict, gbar_values, channel_specs=None):
    """Compute identifiable combinations from recorded gates.

    For NaTa_t: G_eff = gbar_NaTa * m^3 * h
    For K_Tst:  G_eff = gbar_KTst * m^4 * h
    For Ca_HVA: G_eff = gbar_CaHVA * m^2 * h
    etc. (uses CHANNEL_SPECS from config)

    Args:
        gates_dict: dict mapping 'gate_channel_region' to timeseries arrays.
            Example keys: 'm_NaTa_t_soma', 'h_NaTa_t_soma', 'm_Ih_tuft'.
        gbar_values: dict mapping (channel, region) tuples or
            'channel_region' strings to maximal conductance values (S/cm^2).
            If a channel-region pair is missing, gbar defaults to 1.0
            (producing normalised effective conductance).
        channel_specs: override config.CHANNEL_SPECS if needed.
            Format: {channel: {'gates': [letters], 'exp': [ints], 'ion': str}}

    Returns:
        dict mapping 'G_channel_region' to effective conductance timeseries
        (same temporal resolution as input gate arrays).
    """
    specs = channel_specs or CHANNEL_SPECS
    grouped = _group_gates(gates_dict)
    result = {}

    for channel, region_gates in grouped.items():
        if channel not in specs:
            logger.warning("Channel '%s' not in CHANNEL_SPECS, skipping.", channel)
            continue
        spec = specs[channel]
        gate_letters = spec['gates']
        exponents = spec['exp']

        for region, gate_arrays in region_gates.items():
            # Check all required gates are present
            missing = [g for g in gate_letters if g not in gate_arrays]
            if missing:
                logger.warning(
                    "Missing gate(s) %s for %s in %s, skipping.",
                    missing, channel, region,
                )
                continue

            # Look up gbar
            gbar = _lookup_gbar(gbar_values, channel, region)

            # Compute product: gbar * gate_0^exp_0 * gate_1^exp_1 * ...
            g_eff = np.full_like(gate_arrays[gate_letters[0]], gbar, dtype=np.float64)
            for gate_letter, exp in zip(gate_letters, exponents):
                g_eff = g_eff * np.power(gate_arrays[gate_letter].astype(np.float64), exp)

            key = f'G_{channel}_{region}'
            result[key] = g_eff

    logger.info("Computed %d effective conductance timeseries.", len(result))
    return result


def _lookup_gbar(gbar_values, channel, region):
    """Look up maximal conductance from multiple key formats."""
    # Try (channel, region) tuple
    key_tuple = (channel, region)
    if key_tuple in gbar_values:
        return float(gbar_values[key_tuple])

    # Try 'channel_region' string
    key_str = f'{channel}_{region}'
    if key_str in gbar_values:
        return float(gbar_values[key_str])

    # Try channel alone (region-independent)
    if channel in gbar_values:
        return float(gbar_values[channel])

    # Default: normalised (gbar = 1)
    logger.debug("No gbar found for %s/%s, using 1.0.", channel, region)
    return 1.0


# ---------------------------------------------------------------------------
# Ionic current computation
# ---------------------------------------------------------------------------

def compute_ionic_currents(G_eff, voltages, reversal_potentials=None):
    """Compute ionic currents: I_c(t) = G_c(t) * (V(t) - E_c).

    Args:
        G_eff: dict from compute_effective_conductances.
            Keys are 'G_channel_region'.
        voltages: dict mapping region name to voltage timeseries (mV).
            Example: {'soma': array, 'tuft': array}.
        reversal_potentials: dict mapping ion type to E_rev (mV).
            Defaults to config.REVERSAL_POTENTIALS.

    Returns:
        dict mapping 'I_channel_region' to current timeseries.
        Convention: inward current is negative (standard biophysics sign).
    """
    specs = CHANNEL_SPECS
    e_rev = reversal_potentials or REVERSAL_POTENTIALS
    result = {}

    for g_key, g_arr in G_eff.items():
        # Parse 'G_channel_region'
        parts = g_key.split('_', 1)
        if parts[0] != 'G' or len(parts) < 2:
            logger.warning("Unexpected G_eff key format: '%s'", g_key)
            continue

        remainder = parts[1]
        channel, region = _split_channel_region(remainder)
        if channel is None:
            logger.warning("Cannot parse channel/region from '%s'", g_key)
            continue

        if channel not in specs:
            logger.warning("Channel '%s' not in specs for current calc.", channel)
            continue

        ion = specs[channel]['ion']
        if ion not in e_rev:
            logger.warning("No reversal potential for ion '%s'.", ion)
            continue

        if region not in voltages:
            logger.warning("No voltage for region '%s', skipping %s.", region, g_key)
            continue

        V = voltages[region].astype(np.float64)
        E = e_rev[ion]
        current = g_arr * (V - E)

        i_key = f'I_{channel}_{region}'
        result[i_key] = current

    logger.info("Computed %d ionic current timeseries.", len(result))
    return result


def _split_channel_region(remainder):
    """Split 'NaTa_t_soma' into ('NaTa_t', 'soma') using known channels."""
    for channel in CHANNEL_SPECS:
        if remainder.startswith(channel):
            suffix = remainder[len(channel):]
            if suffix.startswith('_') and len(suffix) > 1:
                return channel, suffix[1:]
    return None, None


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

def compute_all(trial_dir=None, save_path=None, gbar_values=None, n_trials=None):
    """Compute Level B targets for all trials and save.

    Loads gate variables and voltages from trial .npz files, computes
    effective conductances and ionic currents, and saves the results.

    Args:
        trial_dir: directory containing trial_XXX.npz files.
            Defaults to config.BAHL_TRIAL_DIR.
        save_path: directory to save Level B arrays.
            Defaults to config.RESULTS_DIR / 'level_B'.
        gbar_values: maximal conductances. If None, uses gbar=1.0 (normalised).
        n_trials: number of trials to process. Defaults to all found.

    Returns:
        Summary dict with variable names and array shapes.
    """
    trial_dir = Path(trial_dir or BAHL_TRIAL_DIR)
    save_path = Path(save_path or RESULTS_DIR / 'level_B')
    save_path.mkdir(parents=True, exist_ok=True)

    gbar_values = gbar_values or {}
    var_names = load_variable_names(trial_dir)

    # Discover available trials
    trial_files = sorted(trial_dir.glob('trial_*.npz'))
    if n_trials is not None:
        trial_files = trial_files[:n_trials]

    if not trial_files:
        logger.error("No trial files found in %s", trial_dir)
        return {}

    logger.info("Processing %d trials from %s", len(trial_files), trial_dir)

    all_cond_keys = None
    all_curr_keys = None

    for i, fpath in enumerate(trial_files):
        idx = int(fpath.stem.split('_')[1])
        trial_data = load_trial(trial_dir, idx)

        # Extract gate variables from Level A data
        gates_dict = _extract_gates(trial_data, var_names)
        voltages = _extract_voltages(trial_data, var_names)

        if not gates_dict:
            logger.warning("No gate variables found in trial %d", idx)
            continue

        # Compute
        G_eff = compute_effective_conductances(gates_dict, gbar_values)
        I_ion = compute_ionic_currents(G_eff, voltages)

        # Collect keys on first iteration
        if all_cond_keys is None:
            all_cond_keys = sorted(G_eff.keys())
            all_curr_keys = sorted(I_ion.keys())

        # Stack into arrays and save
        cond_array = np.column_stack(
            [G_eff[k] for k in all_cond_keys]
        ) if G_eff else np.array([])

        curr_array = np.column_stack(
            [I_ion[k] for k in all_curr_keys]
        ) if I_ion else np.array([])

        np.savez_compressed(
            save_path / f'level_B_{idx:03d}.npz',
            conductances=cond_array,
            currents=curr_array,
        )

    # Save metadata
    summary = {
        'conductance_names': all_cond_keys or [],
        'current_names': all_curr_keys or [],
        'n_trials': len(trial_files),
        'source_dir': str(trial_dir),
    }
    save_results_json(summary, save_path / 'level_B_metadata.json')
    logger.info(
        "Level B complete: %d conductances, %d currents, %d trials.",
        len(all_cond_keys or []),
        len(all_curr_keys or []),
        len(trial_files),
    )
    return summary


def _extract_gates(trial_data, var_names):
    """Extract gate variable timeseries from trial data.

    Expects trial_data to contain 'level_A_gates' array with columns
    corresponding to var_names['level_A'] entries.
    """
    gates = {}
    if 'level_A_gates' not in trial_data:
        return gates

    gate_array = trial_data['level_A_gates']
    if var_names and 'level_A' in var_names:
        names = var_names['level_A']
        for j, name in enumerate(names):
            if j < gate_array.shape[-1]:
                gates[name] = gate_array[..., j]
    return gates


def _extract_voltages(trial_data, var_names):
    """Extract voltage timeseries per region from trial data."""
    voltages = {}
    if 'output' in trial_data:
        # Single output voltage assumed to be soma
        voltages['soma'] = trial_data['output'].flatten()

    # Check for multi-region voltages stored as named variables
    if var_names and 'voltages' in var_names:
        v_array = trial_data.get('voltages', None)
        if v_array is not None:
            for j, region in enumerate(var_names['voltages']):
                if j < v_array.shape[-1]:
                    voltages[region] = v_array[..., j]

    return voltages
