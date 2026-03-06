"""Graded replacement protocol for Phase 3.

Systematically replace L5PCs with surrogates at increasing fractions
and measure degradation across all 4 validation levels.
"""
import numpy as np
from pathlib import Path
from l5pc.config import REPLACEMENT_FRACTIONS, REPLACEMENT_REPLICATES, CIRCUIT_CELL_COUNTS
from l5pc.utils.io import save_results_json


def get_replacement_counts(n_l5pcs, fractions):
    """Convert fractions to actual neuron counts.

    Handles 'single' as a special case (1 neuron).

    Args:
        n_l5pcs: Total number of L5PCs in circuit
        fractions: List of fractions (float) or 'single'

    Returns:
        List of (fraction_label, n_replaced) tuples
    """
    counts = []
    for f in fractions:
        if f == 'single':
            counts.append(('1_cell', 1))
        elif f == 0.0:
            counts.append(('0pct', 0))
        elif f == 1.0:
            counts.append(('100pct', n_l5pcs))
        else:
            n = max(1, int(round(f * n_l5pcs)))
            counts.append((f'{int(f*100)}pct', n))
    return counts


def select_replacement_indices(n_l5pcs, n_replace, replicate_idx, seed=42):
    """Randomly select which L5PCs to replace for a given replicate.

    Args:
        n_l5pcs: Total L5PCs
        n_replace: How many to replace
        replicate_idx: Replicate number (varies random selection)

    Returns:
        Array of L5PC indices to replace
    """
    rng = np.random.RandomState(seed + replicate_idx * 137)
    if n_replace >= n_l5pcs:
        return np.arange(n_l5pcs)
    return rng.choice(n_l5pcs, n_replace, replace=False)


def run_graded_replacement(circuit_builder, surrogate_model, bio_model,
                           n_replicates=None, save_dir=None):
    """Run the full graded replacement protocol.

    Args:
        circuit_builder: Function that builds the circuit with specified
            replacement indices. Signature:
            circuit_builder(replacement_indices) -> circuit_data dict
        surrogate_model: Trained surrogate model
        bio_model: Biological model (for baseline)
        n_replicates: Override config.REPLACEMENT_REPLICATES
        save_dir: Directory to save results

    Returns:
        dict of {fraction_label: {level: {metric: [values]}}}
    """
    from l5pc.validation.level1_output import run_level1_validation
    from l5pc.validation.level2_circuit import run_level2_validation
    from l5pc.validation.level3_descartes import run_level3_validation
    from l5pc.validation.level4_consciousness import run_level4_validation

    if n_replicates is None:
        n_replicates = REPLACEMENT_REPLICATES

    n_l5pcs = CIRCUIT_CELL_COUNTS['L5PC']
    replacement_counts = get_replacement_counts(n_l5pcs, REPLACEMENT_FRACTIONS)

    all_results = {}

    for frac_label, n_replace in replacement_counts:
        print(f"\n=== Replacement: {frac_label} ({n_replace}/{n_l5pcs} L5PCs) ===")

        n_reps = 1 if n_replace == 0 or n_replace == n_l5pcs else n_replicates
        frac_results = {'level1': [], 'level2': [], 'level3': [], 'level4': []}

        for rep in range(n_reps):
            print(f"  Replicate {rep+1}/{n_reps}")

            # Select which cells to replace
            indices = select_replacement_indices(n_l5pcs, n_replace, rep)

            # Build and simulate circuit with hybrid configuration
            try:
                circuit_data = circuit_builder(indices)
            except Exception as e:
                print(f"  ERROR in circuit simulation: {e}")
                continue

            # Run all 4 validation levels
            bio_data = circuit_data.get('bio', {})
            surr_data = circuit_data.get('hybrid', {})

            if bio_data and surr_data:
                try:
                    l1 = run_level1_validation(
                        bio_data.get('outputs', []),
                        surr_data.get('outputs', []),
                        bio_data.get('conditions', [])
                    )
                    frac_results['level1'].append(l1.get('summary', {}))
                except Exception as e:
                    print(f"    Level 1 error: {e}")

                try:
                    l2 = run_level2_validation(bio_data, surr_data)
                    frac_results['level2'].append(l2.get('summary', {}))
                except Exception as e:
                    print(f"    Level 2 error: {e}")

                try:
                    l3 = run_level3_validation(
                        surr_data.get('hidden_states', np.array([])),
                        surr_data.get('bio_targets', {})
                    )
                    frac_results['level3'].append(l3.get('summary', {}))
                except Exception as e:
                    print(f"    Level 3 error: {e}")

                try:
                    l4 = run_level4_validation(bio_data, surr_data)
                    frac_results['level4'].append(l4.get('summary', {}))
                except Exception as e:
                    print(f"    Level 4 error: {e}")

        all_results[frac_label] = frac_results

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_results_json(all_results, save_dir / 'graded_replacement_results.json')

    return all_results


def aggregate_for_plotting(all_results):
    """Reshape results for the plotting functions.

    Returns:
        dict of {level: {metric_name: {fraction: [values]}}}
    """
    plotable = {'level1': {}, 'level2': {}, 'level3': {}, 'level4': {}}

    for frac_label, frac_results in all_results.items():
        for level_key in ['level1', 'level2', 'level3', 'level4']:
            level_reps = frac_results.get(level_key, [])
            for rep_result in level_reps:
                for metric_name, value in rep_result.items():
                    if metric_name not in plotable[level_key]:
                        plotable[level_key][metric_name] = {}
                    if frac_label not in plotable[level_key][metric_name]:
                        plotable[level_key][metric_name][frac_label] = []
                    plotable[level_key][metric_name][frac_label].append(value)

    return plotable
