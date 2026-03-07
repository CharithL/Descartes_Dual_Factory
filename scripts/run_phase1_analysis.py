"""
L5PC DESCARTES -- Phase 1 Post-Pipeline Analysis

Runs all 4 priority analyses on Vast.ai Phase 1 results:
  Priority 1: Progressive ablation on 8 specific MANDATORY targets (h=128)
  Priority 2: Fix BAC detection (logistic regression for binary targets)
  Priority 3: OOD robustness check (Grant et al. 2025)
  Priority 4: Cross-circuit summary table

Usage:
    python scripts/run_phase1_analysis.py --data-dir data/bahl_trials

Outputs:
    data/results/progressive_ablation_h128.json
    data/results/bac_detection_results.json
    data/results/ood_robustness_check.json
    data/results/cross_circuit_summary.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from l5pc.config import (
    ABLATION_K_FRACTIONS,
    ABLATION_N_RANDOM,
    HIDDEN_SIZES,
    RESULTS_DIR,
    SURROGATE_DIR,
)
from l5pc.probing.ablation import (
    _set_model_eval,
    focused_progressive_ablation,
    forward_with_clamp,
    forward_with_resample,
    ood_norm_diagnostic,
    resample_ablation,
    causal_ablation,
    classify_mandatory_type,
    _load_target_for_test,
)
from l5pc.probing.ridge_probe import (
    _load_hidden_states,
    _load_targets,
    logistic_cv_auc,
    probe_binary_variable,
    preprocess,
)
from l5pc.utils.io import (
    load_all_trials,
    load_results_json,
    save_results_json,
)
from l5pc.utils.metrics import cross_condition_correlation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)-20s %(levelname)-5s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('phase1_analysis')


# ===================================================================
# Target specifications for Priority 1
# ===================================================================

PRIORITY1_TARGETS = [
    # (level, var_name, delta_R2 from Vast.ai)
    ('B', 'G_Ih_apical_trunk', 0.692),
    ('B', 'G_Ih_basal',        0.597),
    ('B', 'G_Ih_nexus',        0.563),
    ('B', 'G_Ih_tuft',         0.532),
    ('B', 'G_Ca_HVA_nexus',    0.390),
    ('B', 'G_SKv3_1_nexus',    0.870),
    ('C', 'Ca_hotzone_mean',   0.358),
    ('C', 'cv_isi',            0.296),
]


def _load_model_and_data(data_dir, hidden_dir, model_dir, hs):
    """Load model, hidden states, test inputs/outputs for a given hidden size."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model_path = model_dir / f'lstm_h{hs}_best.pt'
    if not model_path.exists():
        model_path = model_dir / f'lstm_h{hs}.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found for h={hs}")

    from l5pc.surrogates.lstm import L5PC_LSTM
    model = L5PC_LSTM(hidden_size=hs)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    _set_model_eval(model)

    # Load hidden states
    hidden_states = _load_hidden_states(hidden_dir, hs, trained=True)

    # Load test data
    test_trials = load_all_trials(data_dir, split='test')
    if not test_trials:
        raise FileNotFoundError(f"No test trials in {data_dir}")

    test_inputs_np = np.stack([t['inputs'] for t in test_trials])
    test_outputs_np = np.stack([t['output'] for t in test_trials])
    test_inputs = torch.tensor(test_inputs_np, dtype=torch.float32)

    return model, hidden_states, test_inputs, test_outputs_np, test_trials, device


# ===================================================================
# Priority 1: Progressive ablation on 8 specific targets
# ===================================================================

def run_priority1(data_dir, hidden_dir, model_dir, results_dir):
    """Progressive ablation on 8 MANDATORY targets at h=128."""
    logger.info("=" * 60)
    logger.info("PRIORITY 1: Progressive ablation on 8 targets (h=128)")
    logger.info("=" * 60)

    hs = 128
    model, hidden_states, test_inputs, test_outputs, test_trials, device = \
        _load_model_and_data(data_dir, hidden_dir, model_dir, hs)

    all_results = {}

    for level, var_name, expected_delta in PRIORITY1_TARGETS:
        logger.info("\n--- %s (level %s, expected dR2=%.3f) ---",
                    var_name, level, expected_delta)

        target_y = _load_target_for_test(data_dir, level, var_name,
                                          test_trials)
        if target_y is None:
            logger.warning("  Target %s NOT FOUND, skipping", var_name)
            continue

        n = min(len(target_y), hidden_states.shape[0], test_inputs.shape[0])

        result = focused_progressive_ablation(
            model, test_inputs[:n], test_outputs[:n],
            target_y[:n], hidden_states[:n], var_name,
        )
        result['level'] = level
        result['expected_delta_R2'] = expected_delta

        all_results[var_name] = result

        logger.info(
            "  Classification: %s  Breaking point: %s%%",
            result['classification'],
            f"{result['breaking_point']*100:.0f}" if result['breaking_point'] else "N/A",
        )
        logger.info(
            "  High |r| dims (>0.3): %d  Max |r|: %.3f",
            result['n_high_r_dims'],
            result['correlation_stats']['max'],
        )

    # Print comparison table
    _print_comparison_table(all_results)

    # Check Ih gradient uniformity
    _check_ih_gradient(all_results)

    # Save
    save_path = results_dir / 'progressive_ablation_h128.json'
    save_results_json({
        'hidden_size': hs,
        'n_targets': len(all_results),
        'targets': all_results,
    }, save_path)
    logger.info("\nSaved to %s", save_path)
    return all_results


def _print_comparison_table(results):
    """Print the Priority 1 comparison table."""
    logger.info("\n" + "=" * 90)
    logger.info("COMPARISON TABLE: 8 MANDATORY targets at h=128")
    logger.info("=" * 90)

    header = (
        f"{'Variable':<25} {'dR2':>6} {'Type':<22} {'Break':>6} "
        f"{'#r>0.3':>6} {'MaxR':>6} {'BasCC':>6}"
    )
    logger.info(header)
    logger.info("-" * 90)

    for var_name, r in results.items():
        bp = r['breaking_point']
        bp_str = f"{bp*100:.0f}%" if bp else "N/A"
        logger.info(
            "  %-23s %.3f %-22s %6s %6d %.3f %.3f",
            var_name,
            r.get('expected_delta_R2', 0),
            r['classification'],
            bp_str,
            r['n_high_r_dims'],
            r['correlation_stats']['max'],
            r['baseline_cc'],
        )

    logger.info("-" * 90)


def _check_ih_gradient(results):
    """Check whether the Ih gradient breaks uniformly across dendrites."""
    ih_targets = ['G_Ih_apical_trunk', 'G_Ih_basal', 'G_Ih_nexus', 'G_Ih_tuft']
    ih_results = {k: v for k, v in results.items() if k in ih_targets}

    if len(ih_results) < 2:
        return

    logger.info("\n--- Ih gradient uniformity check ---")

    breaking_points = {k: v['breaking_point'] for k, v in ih_results.items()
                       if v['breaking_point'] is not None}
    classifications = {k: v['classification'] for k, v in ih_results.items()}

    # Are all Ih targets the same type?
    unique_types = set(classifications.values())
    if len(unique_types) == 1:
        logger.info("  All 4 Ih targets: %s -- gradient encoded UNIFORMLY",
                     unique_types.pop())
    else:
        logger.info("  Ih targets have MIXED types: %s", classifications)
        logger.info("  -> Gradient may NOT be encoded uniformly")

    if breaking_points:
        bps = list(breaking_points.values())
        bp_strs = {k: f"{v*100:.0f}%" for k, v in breaking_points.items()}
        logger.info("  Breaking points: %s", bp_strs)
        spread = max(bps) - min(bps)
        logger.info("  Spread: %.0f%% -- %s",
                     spread * 100,
                     "tight cluster" if spread < 0.15 else "wide spread")

    # Compare Ca_HVA_nexus concentration
    if 'G_Ca_HVA_nexus' in results:
        ca = results['G_Ca_HVA_nexus']
        ih_max_r = max(v['correlation_stats']['max'] for v in ih_results.values())
        ca_max_r = ca['correlation_stats']['max']
        logger.info("\n  Ca_HVA_nexus max|r|=%.3f vs Ih max|r|=%.3f -- %s",
                     ca_max_r, ih_max_r,
                     "more concentrated" if ca_max_r > ih_max_r
                     else "more distributed")


# ===================================================================
# Priority 2: BAC detection fix
# ===================================================================

def run_priority2(data_dir, hidden_dir, results_dir):
    """Fix BAC detection: logistic regression for binary targets."""
    logger.info("\n" + "=" * 60)
    logger.info("PRIORITY 2: BAC detection (logistic regression)")
    logger.info("=" * 60)

    hs = 128
    trained_H = _load_hidden_states(hidden_dir, hs, trained=True)
    untrained_H = _load_hidden_states(hidden_dir, hs, trained=False)

    targets = _load_targets(data_dir, 'C')
    n = min(trained_H.shape[0], untrained_H.shape[0])

    binary_results = {}

    # Binary targets to probe
    binary_targets = ['bac_detected']
    # Also check n_dendritic_Ca_spikes if it exists (may have been removed)
    if 'n_dendritic_Ca_spikes' in targets:
        binary_targets.append('n_dendritic_Ca_spikes')

    for var_name in binary_targets:
        if var_name not in targets:
            logger.warning("  %s not found in Level C targets", var_name)
            continue

        y = targets[var_name][:n]

        # Force binary for bac_detected
        if var_name == 'bac_detected':
            y = (y > 0.5).astype(float)

        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        logger.info("\n  %s: %d positive, %d negative (total %d)",
                     var_name, n_pos, n_neg, len(y))

        if n_pos == 0:
            logger.warning("  NO positive events -- cannot fit logistic model")
            binary_results[var_name] = {
                'var_name': var_name,
                'metric': 'AUC-ROC',
                'AUC_trained': 0.5,
                'AUC_untrained': 0.5,
                'delta_AUC': 0.0,
                'n_positive': 0,
                'n_negative': n_neg,
                'recommendation': (
                    "Zero BAC events detected. Add BAC-specific simulation "
                    "trials (coincident apical+basal stimulation at 10-20 Hz)."
                ),
            }
            continue

        result = probe_binary_variable(
            trained_H[:n], untrained_H[:n], y, var_name,
        )
        binary_results[var_name] = result

    # Print BAC detection table
    logger.info("\n--- BAC Detection Results ---")
    for var_name, r in binary_results.items():
        logger.info(
            "  %s: AUC_tr=%.3f  AUC_un=%.3f  dAUC=%.3f  "
            "n_pos=%d  n_neg=%d",
            var_name,
            r.get('AUC_trained', 0.5),
            r.get('AUC_untrained', 0.5),
            r.get('delta_AUC', 0.0),
            r.get('n_positive', 0),
            r.get('n_negative', 0),
        )
        if r.get('recommendation'):
            logger.info("  WARNING: %s", r['recommendation'])

    # Save
    save_path = results_dir / 'bac_detection_results.json'
    save_results_json({
        'hidden_size': hs,
        'binary_targets': binary_results,
    }, save_path)
    logger.info("\nSaved to %s", save_path)
    return binary_results


# ===================================================================
# Priority 3: OOD robustness check (Grant et al. 2025)
# ===================================================================

def run_priority3(data_dir, hidden_dir, model_dir, results_dir):
    """OOD robustness: compare mean-clamp vs resample vs random."""
    logger.info("\n" + "=" * 60)
    logger.info("PRIORITY 3: OOD robustness check (Grant et al. 2025)")
    logger.info("=" * 60)

    hs = 128
    model, hidden_states, test_inputs, test_outputs, test_trials, device = \
        _load_model_and_data(data_dir, hidden_dir, model_dir, hs)

    # Focus on G_Ih_apical_trunk as the canonical test case
    var_name = 'G_Ih_apical_trunk'
    level = 'B'

    target_y = _load_target_for_test(data_dir, level, var_name, test_trials)
    if target_y is None:
        logger.error("  Cannot load %s -- aborting Priority 3", var_name)
        return {}

    n = min(len(target_y), hidden_states.shape[0], test_inputs.shape[0])
    target_y = target_y[:n]
    hs_slice = hidden_states[:n]
    ti_slice = test_inputs[:n]
    to_slice = test_outputs[:n]

    # 1. L2 norm diagnostic
    logger.info("\n--- L2 Norm Diagnostic (k=20%%) ---")
    norm_diag = ood_norm_diagnostic(
        model, ti_slice, target_y, hs_slice, k_frac=0.20
    )
    logger.info(
        "  ||h_intact|| = %.3f +/- %.3f",
        norm_diag['norm_intact_mean'], norm_diag['norm_intact_std'],
    )
    logger.info(
        "  ||h_clamped|| = %.3f +/- %.3f",
        norm_diag['norm_clamped_mean'], norm_diag['norm_clamped_std'],
    )
    logger.info(
        "  Ratio: %.3f +/- %.3f  OOD flag: %s",
        norm_diag['ratio_mean'], norm_diag['ratio_std'],
        "YES" if norm_diag['ood_flag'] else "no",
    )

    # 2. Mean-clamp ablation (standard)
    logger.info("\n--- Mean-clamp ablation ---")
    mean_results, mean_baseline = causal_ablation(
        model, ti_slice, to_slice, target_y, hs_slice,
    )

    # 3. Resample ablation (OOD-robust)
    logger.info("\n--- Resample ablation ---")
    resample_results, resample_baseline = resample_ablation(
        model, ti_slice, to_slice, target_y, hs_slice,
    )

    # 4. Comparison table at each k
    logger.info("\n--- OOD Robustness Comparison: %s ---", var_name)
    logger.info(
        "%5s %8s %8s %8s %8s %6s",
        "k%", "MeanCC", "MeanZ", "ResCC", "ResZ", "Agree",
    )
    logger.info("-" * 55)

    comparisons = []
    for m, r in zip(mean_results, resample_results):
        agree = (m['verdict'] == r['verdict'])
        logger.info(
            "  %3.0f%% %8.3f %8.2f %8.3f %8.2f %6s",
            m['k_frac'] * 100,
            m['target_cc'], m['z_score'],
            r['target_cc'], r['z_score'],
            "yes" if agree else "NO",
        )
        comparisons.append({
            'k_frac': m['k_frac'],
            'mean_clamp_cc': m['target_cc'],
            'mean_clamp_z': m['z_score'],
            'mean_clamp_verdict': m['verdict'],
            'resample_cc': r['target_cc'],
            'resample_z': r['z_score'],
            'resample_verdict': r['verdict'],
            'methods_agree': agree,
        })

    # Overall verdict
    all_agree = all(c['methods_agree'] for c in comparisons)
    if all_agree:
        verdict = "ROBUST -- mean-clamp and resample agree at all k"
    elif norm_diag['ood_flag']:
        verdict = "CAUTION -- OOD detected and methods disagree"
    else:
        verdict = "MIXED -- methods disagree but no strong OOD signal"

    logger.info("\nOverall verdict: %s", verdict)

    # Save
    output = {
        'var_name': var_name,
        'hidden_size': hs,
        'norm_diagnostic': norm_diag,
        'mean_clamp': {
            'baseline_cc': mean_baseline,
            'steps': mean_results,
        },
        'resample': {
            'baseline_cc': resample_baseline,
            'steps': resample_results,
        },
        'comparison': comparisons,
        'verdict': verdict,
    }

    save_path = results_dir / 'ood_robustness_check.json'
    save_results_json(output, save_path)
    logger.info("\nSaved to %s", save_path)
    return output


# ===================================================================
# Priority 4: Cross-circuit summary table
# ===================================================================

def run_priority4(results_dir):
    """Cross-circuit summary: hippocampal vs L5PC comparison."""
    logger.info("\n" + "=" * 60)
    logger.info("PRIORITY 4: Cross-circuit summary table")
    logger.info("=" * 60)

    # Load classification summary from Phase 1
    summary_path = results_dir / 'classification_summary.json'
    if not summary_path.exists():
        logger.warning("classification_summary.json not found -- "
                        "computing from available data")
        summary = _compute_summary_from_ablation(results_dir)
    else:
        summary = load_results_json(summary_path)

    # Build the cross-circuit table
    table = {
        'hippocampal_bottleneck': {
            'source': 'Bhalla & Bhatt (2024)',
            'model': 'LSTM h=128',
            'n_variables_probed': 33,
            'n_mandatory': 10,
            'mandatory_fraction': 10 / 33,
            'notable_findings': [
                'g_NMDA_SC: CONCENTRATED (breaks at 3.9%)',
                'gamma_amp: DISTRIBUTED (breaks at 40-60%)',
                'All 3 Ih targets: MANDATORY',
            ],
            'ih_gradient': 'N/A (single compartment)',
        },
        'l5pc': {},
    }

    # Populate L5PC results from Phase 1
    l5pc = {}
    for hs in HIDDEN_SIZES:
        abl_path = results_dir / 'ablation_results.json'
        if not abl_path.exists():
            # Try per-hidden-size file
            abl_path = results_dir / f'ablation_h{hs}.json'
        if not abl_path.exists():
            continue

        data = load_results_json(abl_path)
        results = data.get('results', {})

        n_mandatory = sum(
            1 for r in results.values()
            if r.get('classification', '').startswith('MANDATORY')
        )
        n_total = len(results)

        l5pc[f'h{hs}'] = {
            'n_probed': n_total,
            'n_mandatory': n_mandatory,
            'mandatory_fraction': n_mandatory / max(n_total, 1),
        }

        # Check Ih gradient across hidden sizes
        ih_vars = [k for k in results if 'Ih' in k and f'h{hs}' in k]
        ih_mandatory = sum(
            1 for k in ih_vars
            if results[k].get('classification', '').startswith('MANDATORY')
        )
        l5pc[f'h{hs}']['ih_mandatory'] = ih_mandatory
        l5pc[f'h{hs}']['ih_total'] = len(ih_vars)

    table['l5pc'] = l5pc

    # Also pull from progressive_ablation_h128.json if available
    prog_path = results_dir / 'progressive_ablation_h128.json'
    if prog_path.exists():
        prog_data = load_results_json(prog_path)
        targets = prog_data.get('targets', {})

        ih_breaking = {}
        for vn in ['G_Ih_apical_trunk', 'G_Ih_basal',
                    'G_Ih_nexus', 'G_Ih_tuft']:
            if vn in targets:
                bp = targets[vn].get('breaking_point')
                ih_breaking[vn] = bp

        ih_types = set()
        for vn in ['G_Ih_apical_trunk', 'G_Ih_basal',
                    'G_Ih_nexus', 'G_Ih_tuft']:
            if vn in targets:
                ih_types.add(targets[vn].get('classification'))

        table['ih_gradient_analysis'] = {
            'breaking_points': ih_breaking,
            'uniform': len(ih_types) <= 1,
        }

    # Print summary
    logger.info("\n--- Cross-Circuit Comparison ---")
    logger.info(
        "  %-23s %5s %7s %10s %6s",
        "Circuit", "h", "Probed", "Mandatory", "Frac",
    )
    logger.info("-" * 60)
    logger.info(
        "  %-23s %5s %7s %10s %6s",
        "Hippocampal", "128", "33", "10", "30.3%",
    )
    for hs_key, hs_data in table.get('l5pc', {}).items():
        frac = hs_data.get('mandatory_fraction', 0) * 100
        logger.info(
            "  %-23s %5s %7d %10d %5.1f%%",
            "L5PC", hs_key,
            hs_data.get('n_probed', 0),
            hs_data.get('n_mandatory', 0),
            frac,
        )
    logger.info("-" * 60)

    if 'ih_gradient_analysis' in table:
        ih = table['ih_gradient_analysis']
        logger.info("\nIh gradient breaking points:")
        for vn, bp in ih.get('breaking_points', {}).items():
            logger.info("  %s: %s", vn,
                         f"{bp*100:.0f}%" if bp else "NOT CAUSAL")
        logger.info("  Uniform encoding: %s",
                     "YES" if ih.get('uniform') else "NO")

    # Save
    save_path = results_dir / 'cross_circuit_summary.json'
    save_results_json(table, save_path)
    logger.info("\nSaved to %s", save_path)
    return table


def _compute_summary_from_ablation(results_dir):
    """Fallback: compute classification summary from ablation JSON."""
    abl_path = results_dir / 'ablation_results.json'
    if not abl_path.exists():
        return {}
    return load_results_json(abl_path)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 1 post-pipeline analysis')
    parser.add_argument('--data-dir', type=str, default='data/bahl_trials',
                        help='Directory with trial data')
    parser.add_argument('--hidden-dir', type=str, default=None,
                        help='Directory with hidden states (default: data/surrogates)')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Directory with model checkpoints (default: data/surrogates)')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Results directory (default: data/results)')
    parser.add_argument('--priorities', type=str, default='1,2,3,4',
                        help='Comma-separated priorities to run (default: 1,2,3,4)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    hidden_dir = Path(args.hidden_dir) if args.hidden_dir else SURROGATE_DIR
    model_dir = Path(args.model_dir) if args.model_dir else SURROGATE_DIR
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    priorities = [int(p.strip()) for p in args.priorities.split(',')]

    logger.info("Phase 1 Analysis Configuration:")
    logger.info("  data_dir:    %s", data_dir)
    logger.info("  hidden_dir:  %s", hidden_dir)
    logger.info("  model_dir:   %s", model_dir)
    logger.info("  results_dir: %s", results_dir)
    logger.info("  priorities:  %s", priorities)
    logger.info("")

    if 1 in priorities:
        try:
            run_priority1(data_dir, hidden_dir, model_dir, results_dir)
        except Exception as e:
            logger.error("Priority 1 failed: %s", e, exc_info=True)

    if 2 in priorities:
        try:
            run_priority2(data_dir, hidden_dir, results_dir)
        except Exception as e:
            logger.error("Priority 2 failed: %s", e, exc_info=True)

    if 3 in priorities:
        try:
            run_priority3(data_dir, hidden_dir, model_dir, results_dir)
        except Exception as e:
            logger.error("Priority 3 failed: %s", e, exc_info=True)

    if 4 in priorities:
        try:
            run_priority4(results_dir)
        except Exception as e:
            logger.error("Priority 4 failed: %s", e, exc_info=True)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 analysis complete.")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
