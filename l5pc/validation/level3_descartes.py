"""Level 3 Validation: Mechanistic Correspondence (DESCARTES).

Reuses the probing pipeline from Phase 1 to test whether the surrogate
in the circuit context still encodes the same biological variables.
"""
import numpy as np
from pathlib import Path


def run_level3_validation(surrogate_hidden_states, bio_targets,
                          phase1_results_path=None):
    """Run Ridge ΔR² probing on surrogate hidden states within the circuit.

    This tests whether mandatory variables identified in Phase 1
    are preserved when the surrogate operates in a circuit context.

    Args:
        surrogate_hidden_states: (T, hidden_dim) from surrogate in circuit
        bio_targets: dict of var_name -> (T,) biological target timeseries
        phase1_results_path: Path to Phase 1 classification results

    Returns:
        dict with per-variable R² and comparison to Phase 1
    """
    from l5pc.probing.ridge_probe import probe_single_variable
    from l5pc.config import PREPROCESSING_OPTIONS, RIDGE_ALPHAS, CV_FOLDS

    # Create a dummy "untrained" baseline (random projection of same dim)
    rng = np.random.RandomState(42)
    untrained_H = rng.randn(*surrogate_hidden_states.shape)

    results = {}
    for var_name, target in bio_targets.items():
        result = probe_single_variable(
            surrogate_hidden_states, untrained_H, target, var_name,
            PREPROCESSING_OPTIONS, RIDGE_ALPHAS, CV_FOLDS
        )
        results[var_name] = result

    # Compare to Phase 1 if available
    if phase1_results_path and Path(phase1_results_path).exists():
        from l5pc.utils.io import load_results_json
        phase1 = load_results_json(phase1_results_path)
        for var_name in results:
            if var_name in phase1:
                results[var_name]['phase1_delta_R2'] = phase1[var_name].get('delta_R2', 0)
                results[var_name]['delta_change'] = (
                    results[var_name].get('delta_R2', 0) -
                    phase1[var_name].get('delta_R2', 0)
                )

    # Summary
    n_preserved = sum(
        1 for r in results.values()
        if r.get('delta_R2', 0) > 0.1 and r.get('category') != 'ZOMBIE'
    )
    results['summary'] = {
        'n_variables_tested': len(bio_targets),
        'n_preserved': n_preserved,
        'mean_delta_R2': np.mean([r.get('delta_R2', 0) for r in results.values()
                                  if 'delta_R2' in r]),
    }

    return results
