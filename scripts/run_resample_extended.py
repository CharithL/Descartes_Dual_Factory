"""
L5PC DESCARTES -- Extended Resample Ablation (Zombie Confirmation Test)

The OOD robustness check on G_Ih_apical_trunk (Priority 3) revealed that
ALL mean-clamp mandatory findings at k>=40 percent were artifacts: mean-clamping
pushes hidden states off-manifold, crashing the network to CC around -0.7.
Resample-clamping (which preserves marginal statistics) shows z > -2 at
every k -- meaning Ih is LEARNED but NOT USED.

This script tests whether ANY variable survives resample ablation by
running the identical protocol on 5 additional targets spanning different
channel types, biological roles, and delta-R2 magnitudes.

If ALL show resample z > -2: the L5PC surrogate is a comprehensive
learned zombie.

If ANY shows resample z < -2: that variable is genuinely mandatory.

Usage:
    python scripts/run_resample_extended.py \
        --data-dir data/bahl_trials \
        --hidden-dir data/surrogates/hidden \
        --model-dir data/surrogates \
        --results-dir data/results

Output:
    data/results/resample_ablation_extended.json
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from l5pc.config import (
    ABLATION_K_FRACTIONS,
    ABLATION_N_RANDOM,
    CAUSAL_Z_THRESHOLD,
)
from l5pc.probing.ablation import (
    _set_model_eval,
    _load_target_for_test,
    causal_ablation,
    resample_ablation,
)
from l5pc.probing.ridge_probe import _load_hidden_states
from l5pc.utils.io import load_all_trials, save_results_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)-20s %(levelname)-5s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('resample_extended')


EXTENDED_TARGETS = [
    ('B', 'G_Ca_HVA_nexus',  0.390),
    ('B', 'G_SKv3_1_nexus',  0.870),
    ('C', 'Ca_hotzone_mean', 0.358),
    ('B', 'G_Im_nexus',      0.247),
    ('B', 'G_SK_E2_nexus',   0.140),
]

IH_BASELINE = {
    'var_name': 'G_Ih_apical_trunk',
    'delta_R2': 0.692,
    'max_abs_z': 1.10,
    'any_causal': False,
    'verdict': 'LEARNED ZOMBIE',
}


def _load_model_and_data(data_dir, hidden_dir, model_dir, hs):
    """Load model, hidden states, test inputs/outputs for a given hidden size."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = model_dir / 'lstm_h{}_best.pt'.format(hs)
    if not model_path.exists():
        model_path = model_dir / 'lstm_h{}.pt'.format(hs)
    if not model_path.exists():
        raise FileNotFoundError('Model not found for h={}'.format(hs))

    from l5pc.surrogates.lstm import L5PC_LSTM
    model = L5PC_LSTM(hidden_size=hs)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    _set_model_eval(model)

    hidden_states = _load_hidden_states(hidden_dir, hs, trained=True)

    test_trials = load_all_trials(data_dir, split='test')
    if not test_trials:
        raise FileNotFoundError('No test trials in {}'.format(data_dir))

    test_inputs_np = np.stack([t['inputs'] for t in test_trials])
    test_outputs_np = np.stack([t['output'] for t in test_trials])
    test_inputs = torch.tensor(test_inputs_np, dtype=torch.float32)

    return model, hidden_states, test_inputs, test_outputs_np, test_trials, device


def run_single_target(model, test_inputs, test_outputs, target_y,
                      hidden_states, var_name, level, delta_r2):
    """Run full resample ablation + mean-clamp at k=40 percent for one target."""
    n = min(len(target_y), hidden_states.shape[0], test_inputs.shape[0])
    ty = target_y[:n]
    hs = hidden_states[:n]
    ti = test_inputs[:n]
    to = test_outputs[:n]

    logger.info('  Running resample ablation...')
    resample_results, resample_baseline = resample_ablation(
        model, ti, to, ty, hs,
    )

    logger.info('  Running mean-clamp at k=40 percent only...')
    mean_results_40, mean_baseline = causal_ablation(
        model, ti, to, ty, hs,
        k_fractions=[0.40],
        n_random_repeats=ABLATION_N_RANDOM,
    )

    resample_40 = None
    for r in resample_results:
        if abs(r['k_frac'] - 0.40) < 0.001:
            resample_40 = r
            break

    all_z = [abs(r['z_score']) for r in resample_results]
    max_abs_z = max(all_z)
    any_causal = any(r['verdict'] == 'CAUSAL' for r in resample_results)

    if any_causal:
        causal_ks = [r['k_frac'] for r in resample_results
                     if r['verdict'] == 'CAUSAL']
        verdict = 'MANDATORY (causal at k={})'.format(causal_ks)
    else:
        verdict = 'LEARNED ZOMBIE'

    return {
        'var_name': var_name,
        'level': level,
        'delta_R2': delta_r2,
        'resample_baseline_cc': resample_baseline,
        'resample_steps': resample_results,
        'mean_clamp_40': mean_results_40[0] if mean_results_40 else None,
        'resample_40': resample_40,
        'max_abs_z_resample': max_abs_z,
        'any_causal': any_causal,
        'verdict': verdict,
    }


def print_target_table(var_name, resample_results):
    """Print per-target resample ablation table."""
    logger.info('')
    logger.info('  %s:', var_name)
    logger.info(
        '  %5s | %11s | %10s | %8s | %s',
        'k pct', 'Resample CC', 'Random CC', 'z-score', 'Verdict',
    )
    logger.info('  %s', '-' * 60)
    for r in resample_results:
        logger.info(
            '  %4.0f%% | %11.3f | %10.3f | %8.2f | %s',
            r['k_frac'] * 100,
            r['target_cc'],
            r['random_cc_mean'],
            r['z_score'],
            r['verdict'],
        )


def print_summary_table(all_results):
    """Print the final zombie classification summary."""
    logger.info('')
    logger.info('=' * 80)
    logger.info('RESAMPLE ABLATION SUMMARY -- Zombie Classification')
    logger.info('=' * 80)

    logger.info(
        '  %-22s | %7s | %11s | %13s | %s',
        'Variable', 'dR2', 'Max |z| res', 'Any k causal?', 'Verdict',
    )
    logger.info('  %s', '-' * 80)

    logger.info(
        '  %-22s | %7.3f | %11.2f | %13s | %s',
        IH_BASELINE['var_name'],
        IH_BASELINE['delta_R2'],
        IH_BASELINE['max_abs_z'],
        'NO',
        IH_BASELINE['verdict'],
    )

    n_zombie = 1
    n_mandatory = 0

    for r in all_results:
        causal_str = 'YES' if r['any_causal'] else 'NO'
        logger.info(
            '  %-22s | %7.3f | %11.2f | %13s | %s',
            r['var_name'],
            r['delta_R2'],
            r['max_abs_z_resample'],
            causal_str,
            r['verdict'],
        )
        if r['any_causal']:
            n_mandatory += 1
        else:
            n_zombie += 1

    logger.info('  %s', '-' * 80)

    total = n_zombie + n_mandatory
    if n_mandatory == 0:
        overall = 'ALL ZOMBIE -- L5PC surrogate is a comprehensive learned zombie'
    elif n_zombie == 0:
        overall = 'ALL MANDATORY -- OOD artifact was limited to Ih only'
    else:
        overall = 'MIXED -- {}/{} zombie, {}/{} mandatory'.format(
            n_zombie, total, n_mandatory, total,
        )

    logger.info('  OVERALL: %s', overall)
    return overall


def print_comparison_at_40(all_results):
    """Print mean-clamp vs resample comparison at k=40 percent."""
    logger.info('')
    logger.info('=' * 80)
    logger.info('MEAN-CLAMP vs RESAMPLE at k=40 pct (OOD artifact check)')
    logger.info('=' * 80)

    logger.info(
        '  %-22s | %13s | %11s | %8s | %10s',
        'Variable', 'Mean-clamp CC', 'Resample CC', 'Mean z', 'Resample z',
    )
    logger.info('  %s', '-' * 80)

    for r in all_results:
        mc = r.get('mean_clamp_40')
        rs = r.get('resample_40')

        mc_cc = mc['target_cc'] if mc else float('nan')
        mc_z = mc['z_score'] if mc else float('nan')
        rs_cc = rs['target_cc'] if rs else float('nan')
        rs_z = rs['z_score'] if rs else float('nan')

        logger.info(
            '  %-22s | %13.3f | %11.3f | %8.2f | %10.2f',
            r['var_name'], mc_cc, rs_cc, mc_z, rs_z,
        )

    all_disagree = True
    for r in all_results:
        mc = r.get('mean_clamp_40')
        rs = r.get('resample_40')
        if mc and rs:
            if mc['verdict'] == rs['verdict']:
                all_disagree = False

    if all_disagree:
        logger.info('')
        logger.info(
            '  CONCLUSION: Mean-clamp shows CAUSAL but resample shows '
            'NON_CAUSAL for ALL targets at k=40 pct.'
        )
        logger.info(
            '  The OOD artifact is UNIVERSAL. Mean-clamp ablation is '
            'unreliable for this LSTM.'
        )
    else:
        logger.info('')
        logger.info(
            '  Note: Some targets AGREE between methods -- '
            'OOD artifact may not be universal.'
        )


def main():
    parser = argparse.ArgumentParser(
        description='Extended resample ablation for zombie confirmation',
    )
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--hidden-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path, required=True)
    args = parser.parse_args()

    logger.info('=' * 70)
    logger.info('EXTENDED RESAMPLE ABLATION -- Zombie Confirmation Test')
    logger.info('=' * 70)
    logger.info(
        'Testing 5 targets to confirm/refute learned zombie hypothesis',
    )
    logger.info(
        'Targets: %s', [t[1] for t in EXTENDED_TARGETS],
    )

    hs = 128
    model, hidden_states, test_inputs, test_outputs, test_trials, device = \
        _load_model_and_data(args.data_dir, args.hidden_dir, args.model_dir, hs)

    all_results = []

    for level, var_name, delta_r2 in EXTENDED_TARGETS:
        logger.info('')
        logger.info('=' * 60)
        logger.info(
            'TARGET: %s (level %s, dR2=%.3f)', var_name, level, delta_r2,
        )
        logger.info('=' * 60)

        target_y = _load_target_for_test(
            args.data_dir, level, var_name, test_trials,
        )
        if target_y is None:
            logger.warning('  Target %s NOT FOUND, skipping', var_name)
            continue

        result = run_single_target(
            model, test_inputs, test_outputs, target_y,
            hidden_states, var_name, level, delta_r2,
        )
        all_results.append(result)

        print_target_table(var_name, result['resample_steps'])

    overall_verdict = print_summary_table(all_results)

    print_comparison_at_40(all_results)

    save_path = args.results_dir / 'resample_ablation_extended.json'
    save_results_json({
        'hidden_size': hs,
        'n_targets_tested': len(all_results),
        'ih_baseline': IH_BASELINE,
        'targets': {r['var_name']: r for r in all_results},
        'overall_verdict': overall_verdict,
        'causal_z_threshold': CAUSAL_Z_THRESHOLD,
        'k_fractions': list(ABLATION_K_FRACTIONS),
        'n_random_repeats': ABLATION_N_RANDOM,
    }, save_path)
    logger.info('')
    logger.info('Saved to %s', save_path)


if __name__ == '__main__':
    main()
