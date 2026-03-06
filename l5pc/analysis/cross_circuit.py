"""
Cross-circuit comparison table: unified view of DESCARTES results.

Generates a comparison table across all circuit experiments in the
DESCARTES project (thalamic, hippocampal, L5PC Bahl, L5PC Hay),
summarising variable counts, probing results by level, and mandatory
variable types discovered at each level.
"""
import json
import logging
from pathlib import Path

import numpy as np

from l5pc.config import (
    RESULTS_DIR,
    BAHL_TRIAL_DIR,
    HAY_TRIAL_DIR,
    CHANNEL_SPECS,
    BAHL_REGIONS,
    HAY_N_COMPARTMENTS,
    N_BASAL_SYN,
    N_APICAL_SYN,
    N_SOMA_SYN,
)
from l5pc.utils.io import load_results_json, save_results_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference data for prior circuits (from published experiments)
# ---------------------------------------------------------------------------

_THALAMIC_REFERENCE = {
    'circuit': 'Thalamic',
    'total_vars': 160,
    'level_A': {'tested': 0, 'total': 160, 'label': '0/160'},
    'level_B': {'tested': None, 'total': None, 'label': 'NOT TESTED'},
    'level_C': {'tested': None, 'total': None, 'label': 'NOT TESTED'},
    'mandatory_type': 'Unknown',
    'notes': 'Zombie surrogate passed all output tests; '
             'no internal probing was performed.',
}

_HIPPOCAMPAL_REFERENCE = {
    'circuit': 'Hippocampal',
    'total_vars': 25,
    'level_A': {'tested': 0, 'total': 25, 'label': '0/25'},
    'level_B': {'tested': None, 'total': None, 'label': 'N/A'},
    'level_C': {'tested': 2, 'total': 10, 'label': '2/~10'},
    'mandatory_type': 'gamma + NMDA',
    'notes': 'gamma_amp and NMDA current were mandatory non-zombie variables. '
             'Cross-condition correlation was the key metric.',
}


# ---------------------------------------------------------------------------
# L5PC variable counts (estimated from model structure)
# ---------------------------------------------------------------------------

def _estimate_bahl_var_counts():
    """Estimate variable counts for the Bahl reduced L5PC model.

    Level A (raw gates): each channel has gates in each region where
    it is expressed. Rough estimate based on 10 channels x ~3 regions
    x ~1.5 gates per channel = ~45 gate variables, plus voltages.

    Level B (effective conductances + currents): ~10 channels x ~3 regions
    = ~30 conductances + ~30 currents = ~60 variables.

    Level C (emergent properties): ~12 scalar/timeseries observables.
    """
    n_channels = len(CHANNEL_SPECS)
    avg_regions = 3  # Not all channels in all regions
    avg_gates = sum(len(s['gates']) for s in CHANNEL_SPECS.values()) / n_channels

    level_a_gates = int(n_channels * avg_regions * avg_gates)
    level_a_voltages = len(BAHL_REGIONS)
    level_a_total = level_a_gates + level_a_voltages

    level_b_cond = int(n_channels * avg_regions)
    level_b_curr = level_b_cond
    level_b_total = level_b_cond + level_b_curr

    level_c_total = 12  # See emergent_properties.py

    total = level_a_total + level_b_total + level_c_total

    return {
        'total': total,
        'level_A_total': level_a_total,
        'level_B_total': level_b_total,
        'level_C_total': level_c_total,
    }


def _estimate_hay_var_counts():
    """Estimate variable counts for the Hay detailed L5PC model.

    The Hay model has 639 compartments with potentially different
    channel complements. Total variables scale dramatically:
    ~639 compartments x ~8 active channels x ~1.5 gates = ~7,672 gates.
    """
    n_compartments = HAY_N_COMPARTMENTS
    n_channels = len(CHANNEL_SPECS)
    avg_gates = sum(len(s['gates']) for s in CHANNEL_SPECS.values()) / n_channels
    channel_fraction = 0.6  # Not all channels in all compartments

    level_a_gates = int(n_compartments * n_channels * avg_gates * channel_fraction)
    level_a_voltages = n_compartments
    level_a_total = level_a_gates + level_a_voltages

    level_b_cond = int(n_compartments * n_channels * channel_fraction)
    level_b_curr = level_b_cond
    level_b_total = level_b_cond + level_b_curr

    level_c_total = 12  # Same emergent properties

    total = level_a_total + level_b_total + level_c_total

    return {
        'total': total,
        'level_A_total': level_a_total,
        'level_B_total': level_b_total,
        'level_C_total': level_c_total,
    }


# ---------------------------------------------------------------------------
# Load actual experiment results
# ---------------------------------------------------------------------------

def load_circuit_results():
    """Load results from thalamic, hippocampal, and L5PC experiments.

    Returns a list of dicts, one per circuit, each containing:
    - circuit: name
    - total_vars: total number of biophysical variables
    - level_A, level_B, level_C: dicts with 'tested', 'total', 'label'
    - mandatory_type: string describing discovered mandatory variables
    - notes: additional context

    Prior circuit results (thalamic, hippocampal) use hardcoded reference
    values from the published experiments. L5PC results are loaded from
    the results directory if available, otherwise estimated counts are used.
    """
    results = [
        _THALAMIC_REFERENCE.copy(),
        _HIPPOCAMPAL_REFERENCE.copy(),
    ]

    # Bahl model results
    bahl = _load_l5pc_results('L5PC (Bahl)', _estimate_bahl_var_counts())
    results.append(bahl)

    # Hay model results
    hay = _load_l5pc_results('L5PC (Hay)', _estimate_hay_var_counts())
    results.append(hay)

    return results


def _load_l5pc_results(circuit_name, var_counts):
    """Load L5PC results from disk or use estimates.

    Attempts to load probing results from the results directory.
    Falls back to estimated variable counts with TBD labels.
    """
    entry = {
        'circuit': circuit_name,
        'total_vars': var_counts['total'],
        'mandatory_type': 'TBD',
        'notes': '',
    }

    # Try loading actual probing results
    probing_path = RESULTS_DIR / 'probing'
    level_b_path = RESULTS_DIR / 'level_B'
    level_c_path = RESULTS_DIR / 'level_C'

    # Level A: gate variable probing
    level_a_results = _try_load_level_results(probing_path / 'level_A_results.json')
    if level_a_results:
        n_tested = level_a_results.get('n_tested', 0)
        n_passed = level_a_results.get('n_non_zombie', '?')
        entry['level_A'] = {
            'tested': n_passed,
            'total': var_counts['level_A_total'],
            'label': f'{n_passed}/{var_counts["level_A_total"]}',
        }
        mandatory = level_a_results.get('mandatory_variables', [])
        if mandatory:
            entry['mandatory_type'] = ', '.join(mandatory[:5])
            entry['notes'] = f'{len(mandatory)} mandatory variables found at Level A.'
    else:
        entry['level_A'] = {
            'tested': '?',
            'total': var_counts['level_A_total'],
            'label': f'?/~{var_counts["level_A_total"]}',
        }

    # Level B: effective conductance probing
    level_b_results = _try_load_level_results(probing_path / 'level_B_results.json')
    if level_b_results:
        n_passed = level_b_results.get('n_non_zombie', '?')
        entry['level_B'] = {
            'tested': n_passed,
            'total': var_counts['level_B_total'],
            'label': f'{n_passed}/{var_counts["level_B_total"]}',
        }
    else:
        entry['level_B'] = {
            'tested': '?',
            'total': var_counts['level_B_total'],
            'label': f'?/~{var_counts["level_B_total"]}',
        }

    # Level C: emergent properties probing
    level_c_results = _try_load_level_results(probing_path / 'level_C_results.json')
    if level_c_results:
        n_passed = level_c_results.get('n_non_zombie', '?')
        entry['level_C'] = {
            'tested': n_passed,
            'total': var_counts['level_C_total'],
            'label': f'{n_passed}/{var_counts["level_C_total"]}',
        }
    else:
        entry['level_C'] = {
            'tested': '?',
            'total': var_counts['level_C_total'],
            'label': f'?/~{var_counts["level_C_total"]}',
        }

    return entry


def _try_load_level_results(path):
    """Attempt to load results JSON, return None on failure."""
    path = Path(path)
    if path.exists():
        try:
            return load_results_json(path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
    return None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_HEADER = (
    f"{'Circuit':<16} | {'Total vars':>10} | {'Level A':>12} | "
    f"{'Level B':>16} | {'Level C':>16} | {'Mandatory type':<20}"
)
_SEPARATOR = '-' * len(_HEADER)


def print_comparison():
    """Print formatted cross-circuit comparison table.

    Circuit          | Total vars | Level A      | Level B          | Level C           | Mandatory type
    -----------------------------------------------------------------------------------------------------------------
    Thalamic         |        160 |        0/160 |       NOT TESTED |       NOT TESTED  | Unknown
    Hippocampal      |         25 |         0/25 |              N/A |           2/~10   | gamma + NMDA
    L5PC (Bahl)      |        ~90 |       ?/~30  |          ?/~25   |           ?/~12   | TBD
    L5PC (Hay)       |     ~5,000 |    ?/~2,500  |       ?/~1,000   |           ?/~12   | TBD
    """
    results = load_circuit_results()

    print()
    print("=" * len(_HEADER))
    print("  DESCARTES Cross-Circuit Comparison")
    print("=" * len(_HEADER))
    print()
    print(_HEADER)
    print(_SEPARATOR)

    for r in results:
        total_str = _format_total(r['total_vars'])
        line = (
            f"{r['circuit']:<16} | {total_str:>10} | "
            f"{r['level_A']['label']:>12} | "
            f"{r['level_B']['label']:>16} | "
            f"{r['level_C']['label']:>16} | "
            f"{r['mandatory_type']:<20}"
        )
        print(line)

    print(_SEPARATOR)
    print()

    # Print notes
    notes = [(r['circuit'], r['notes']) for r in results if r.get('notes')]
    if notes:
        print("Notes:")
        for circuit, note in notes:
            print(f"  {circuit}: {note}")
        print()


def _format_total(n):
    """Format total variable count with ~ prefix for estimates."""
    if n > 1000:
        return f'~{n:,}'
    elif n > 50:
        return f'~{n}'
    else:
        return str(n)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_comparison_table(save_path=None):
    """Save comparison as JSON and formatted text.

    Args:
        save_path: directory to save files. Defaults to config.RESULTS_DIR.

    Saves:
        - cross_circuit_comparison.json: structured data
        - cross_circuit_comparison.txt: formatted text table
    """
    save_path = Path(save_path or RESULTS_DIR)
    save_path.mkdir(parents=True, exist_ok=True)

    results = load_circuit_results()

    # JSON output (with numpy conversion)
    json_data = {
        'description': 'DESCARTES cross-circuit comparison table',
        'circuits': results,
        'level_definitions': {
            'Level A': 'Raw biophysical variables (gates, concentrations)',
            'Level B': 'Identifiable combinations (effective conductances, '
                       'ionic currents)',
            'Level C': 'Emergent functional properties (burst ratio, BAC index, '
                       'Ca spike amplitude)',
        },
        'interpretation': {
            'label_format': 'non_zombie_count / total_tested',
            'TBD': 'Experiment not yet run',
            'NOT TESTED': 'Level not probed in original experiment',
            'N/A': 'Level not applicable to this circuit',
        },
    }
    save_results_json(json_data, save_path / 'cross_circuit_comparison.json')

    # Text output
    text_lines = []
    text_lines.append("DESCARTES Cross-Circuit Comparison")
    text_lines.append("=" * 80)
    text_lines.append("")
    text_lines.append(_HEADER)
    text_lines.append(_SEPARATOR)

    for r in results:
        total_str = _format_total(r['total_vars'])
        line = (
            f"{r['circuit']:<16} | {total_str:>10} | "
            f"{r['level_A']['label']:>12} | "
            f"{r['level_B']['label']:>16} | "
            f"{r['level_C']['label']:>16} | "
            f"{r['mandatory_type']:<20}"
        )
        text_lines.append(line)

    text_lines.append(_SEPARATOR)
    text_lines.append("")

    notes = [(r['circuit'], r['notes']) for r in results if r.get('notes')]
    if notes:
        text_lines.append("Notes:")
        for circuit, note in notes:
            text_lines.append(f"  {circuit}: {note}")
        text_lines.append("")

    text_content = '\n'.join(text_lines)
    text_path = save_path / 'cross_circuit_comparison.txt'
    with open(text_path, 'w') as f:
        f.write(text_content)

    logger.info("Saved comparison table to %s", save_path)
    return json_data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print_comparison()
    save_comparison_table()
