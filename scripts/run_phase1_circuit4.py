#!/usr/bin/env python3
"""
DESCARTES Dual Factory v3.0 -- Phase 1 Analysis: Circuit 4 (Human MTL->Frontal)

Runs the hardened battery on pre-extracted hidden states for all human WM
patients. Uses both bin-level (statistical power) and window-level (clean
independence) resolutions. The frequency-resolved R2 tells us which timescale
the encoding lives at.

For each patient:
  1. Bin-level hardening  (n_test * T samples, 20 Hz sampling)
  2. Window-level hardening (n_test samples, independent trials)
  3. SAE superposition detection (bin-level)
  4. MLP nonlinear encoding check (bin-level)

Pre-requisite: Run extract_circuit4_hidden_states.py locally first,
then upload circuit4_extracted/ to Vast.ai.

Usage:
    python scripts/run_phase1_circuit4.py [--patient SUB_ID] [--analysis 0|1|2|3]
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

C4_DIR = PROJECT_ROOT / 'circuit4_extracted'
RESULTS_DIR = PROJECT_ROOT / 'data' / 'results' / 'circuit4'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / 'phase1_circuit4.log'),
    ],
)
logger = logging.getLogger('phase1_c4')


# =========================================================================
# Data Loading
# =========================================================================

def load_patient(patient_dir):
    """Load pre-extracted data for one patient.

    Returns dict with bin/win hidden states, targets, and metadata.
    """
    pd = Path(patient_dir)
    with open(pd / 'patient_meta.json') as f:
        meta = json.load(f)
    with open(pd / 'probe_target_names.json') as f:
        target_names = json.load(f)

    return {
        'meta': meta,
        'target_names': target_names,
        'h_bin_trained': np.load(pd / 'hidden_bin_trained.npy'),
        'h_bin_untrained': np.load(pd / 'hidden_bin_untrained.npy'),
        'h_win_trained': np.load(pd / 'hidden_win_trained.npy'),
        'h_win_untrained': np.load(pd / 'hidden_win_untrained.npy'),
        'targets_bin': np.load(pd / 'probe_targets_bin.npy'),
        'targets_win': np.load(pd / 'probe_targets_win.npy'),
    }


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


# =========================================================================
# Analysis 1: Hardened Probe (bin-level + window-level)
# =========================================================================

def run_hardening_patient(data):
    """Run hardened_probe at both resolutions for one patient."""
    from l5pc.probing.hardening import hardened_probe

    meta = data['meta']
    patient_id = meta['patient_id']
    target_names = data['target_names']
    sampling_rate = meta['sampling_rate_hz']

    device = 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
    except ImportError:
        pass

    results = {'bin_level': {}, 'window_level': {}}

    # -- Bin-level hardening --
    logger.info("  [BIN-LEVEL] %d samples, %d targets, sr=%.0f Hz",
                data['h_bin_trained'].shape[0], len(target_names), sampling_rate)

    for j, name in enumerate(target_names):
        target = data['targets_bin'][:, j].astype(np.float64)
        if target.std() < 1e-10:
            results['bin_level'][name] = {'skipped': True, 'reason': 'zero_variance'}
            continue

        logger.info("    [%d/%d] %s (bin) ...", j + 1, len(target_names), name)
        t1 = time.time()

        # For bin-level, we need an input signal for frequency analysis.
        # Use the trained hidden states' first PC as proxy input.
        # This is a reasonable proxy since we don't have raw inputs on Vast.ai.
        input_proxy = data['h_bin_trained'][:, 0:1].astype(np.float64)

        result = hardened_probe(
            hidden_trained=data['h_bin_trained'].astype(np.float64),
            hidden_untrained=data['h_bin_untrained'].astype(np.float64),
            target=target,
            target_name=name,
            input_signal=input_proxy,
            sampling_rate=sampling_rate,
            device=device,
        )
        elapsed = time.time() - t1
        results['bin_level'][name] = result

        logger.info("      verdict=%s  ridge_dR2=%.4f  mlp_dR2=%.4f  "
                     "p_block=%.4f  BF01=%.2f  DW=%.2f  [%.1fs]",
                     result['hardened_verdict'], result['ridge_delta_r2'],
                     result['mlp_delta_r2'], result['p_block_permutation'],
                     result['bayes_factor']['bf01'], result['durbin_watson'], elapsed)

    # -- Window-level hardening --
    n_win = data['h_win_trained'].shape[0]
    logger.info("  [WINDOW-LEVEL] %d samples (independent trials), %d targets",
                n_win, len(target_names))

    # Window-level: only run if we have enough samples
    if n_win >= 20:
        for j, name in enumerate(target_names):
            target = data['targets_win'][:, j].astype(np.float64)
            if target.std() < 1e-10:
                results['window_level'][name] = {'skipped': True, 'reason': 'zero_variance'}
                continue

            logger.info("    [%d/%d] %s (win) ...", j + 1, len(target_names), name)
            t1 = time.time()

            # Window-level: no temporal structure, use dummy input
            input_proxy = data['h_win_trained'][:, 0:1].astype(np.float64)

            result = hardened_probe(
                hidden_trained=data['h_win_trained'].astype(np.float64),
                hidden_untrained=data['h_win_untrained'].astype(np.float64),
                target=target,
                target_name=name,
                input_signal=input_proxy,
                sampling_rate=1.0,  # 1 sample per trial, no temporal structure
                device=device,
            )
            elapsed = time.time() - t1
            results['window_level'][name] = result

            logger.info("      verdict=%s  ridge_dR2=%.4f  mlp_dR2=%.4f  "
                         "p_block=%.4f  BF01=%.2f  DW=%.2f  [%.1fs]",
                         result['hardened_verdict'], result['ridge_delta_r2'],
                         result['mlp_delta_r2'], result['p_block_permutation'],
                         result['bayes_factor']['bf01'], result['durbin_watson'], elapsed)
    else:
        logger.info("  SKIP window-level: n=%d < 20 (insufficient for hardening)", n_win)
        for name in target_names:
            results['window_level'][name] = {
                'skipped': True,
                'reason': f'insufficient_samples_n={n_win}',
            }

    return results


# =========================================================================
# Analysis 2: SAE Superposition Detection (bin-level)
# =========================================================================

def run_sae_patient(data):
    """Run SAE superposition check on bin-level hidden states."""
    from l5pc.probing.sae_probe import train_sae, sae_probe_biological_variables

    meta = data['meta']
    target_names = data['target_names']
    T = meta['T_bins_per_trial']
    n_test = meta['n_test_trials']
    hidden_size = meta['hidden_size']

    device = 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
    except ImportError:
        pass

    # Reshape to trial lists for train_sae
    h_trained_3d = data['h_bin_trained'].reshape(n_test, T, hidden_size)
    targets_3d = data['targets_bin'].reshape(n_test, T, -1)

    hidden_list = [h_trained_3d[i] for i in range(n_test)]
    bio_list = [targets_3d[i] for i in range(n_test)]

    results = {}
    for exp_factor in [4, 8]:
        logger.info("    SAE: expansion=%dx, k=20, epochs=50", exp_factor)
        t0 = time.time()

        sae, loss_history = train_sae(
            hidden_list, hidden_size,
            expansion_factor=exp_factor, k=20,
            epochs=50, batch_size=4096, device=device,
        )
        train_time = time.time() - t0

        probe_result = sae_probe_biological_variables(
            sae, hidden_list, bio_list, target_names, device=device,
        )

        key = f'exp{exp_factor}'
        results[key] = {
            'expansion_factor': exp_factor,
            'final_loss': float(loss_history[-1]),
            'train_time_s': round(train_time, 1),
            'n_alive': probe_result['n_alive'],
            'mean_monosemanticity': probe_result['mean_monosemanticity'],
            'sae_r2': {k: float(v) for k, v in probe_result['sae_r2'].items()},
            'raw_r2': {k: float(v) for k, v in probe_result['raw_r2'].items()},
            'superposition_detected': probe_result['superposition_detected'],
        }

        total_feats = exp_factor * hidden_size
        logger.info("      Alive: %d/%d  Mono: %.4f",
                     probe_result['n_alive'], total_feats,
                     probe_result['mean_monosemanticity'])

        for name in target_names:
            raw = probe_result['raw_r2'][name]
            sae_r = probe_result['sae_r2'][name]
            sup = probe_result['superposition_detected'][name]
            logger.info("      %s: raw=%.4f sae=%.4f %s",
                         name, raw, sae_r, "SUPERPOSED" if sup else "")

    return results


# =========================================================================
# Analysis 3: MLP Nonlinear Encoding Check (bin-level)
# =========================================================================

def run_mlp_patient(data):
    """Run MLP delta-R2 on bin-level hidden states."""
    from l5pc.probing.mlp_probe import mlp_delta_r2

    target_names = data['target_names']

    device = 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
    except ImportError:
        pass

    results = mlp_delta_r2(
        data['h_bin_trained'].astype(np.float32),
        data['h_bin_untrained'].astype(np.float32),
        data['targets_bin'].astype(np.float32),
        target_names, hidden_dim=64, epochs=50, lr=1e-3,
        n_splits=5, device=device,
    )

    for name in target_names:
        r = results[name]
        logger.info("    %s: ridge_dR2=%.4f  mlp_dR2=%.4f  gain=%.4f  [%s]",
                     name, r['ridge_delta'], r['mlp_delta'],
                     r['nonlinear_gain'], r['encoding_type'])

    return results


# =========================================================================
# Main
# =========================================================================

def print_hardening_table(patient_id, results, level):
    """Print hardening results for one level."""
    data = results.get(level, {})
    if not data:
        return

    print(f"\n{'='*90}")
    print(f"HARDENING -- {patient_id} ({level})")
    print(f"{'='*90}")
    print(f"{'Variable':<25} {'Verdict':<30} {'Ridge dR2':>10} {'MLP dR2':>10} "
          f"{'p_block':>8} {'BF01':>8}")
    print("-" * 90)
    for name, r in data.items():
        if r.get('skipped'):
            print(f"{name:<25} {'SKIPPED':<30} {'':>10} {'':>10} {'':>8} {'':>8}")
            continue
        print(f"{name:<25} {r['hardened_verdict']:<30} "
              f"{r['ridge_delta_r2']:>10.4f} {r['mlp_delta_r2']:>10.4f} "
              f"{r['p_block_permutation']:>8.4f} {r['bayes_factor']['bf01']:>8.2f}")
    print("=" * 90)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='DESCARTES Phase 1 -- Circuit 4')
    parser.add_argument('--patient', type=str, default=None,
                        help='Run specific patient only (e.g. sub-1_ses-1_ecephys+image)')
    parser.add_argument('--analysis', type=int, default=0,
                        help='1=hardening, 2=SAE, 3=MLP, 0=all')
    args = parser.parse_args()

    logger.info("DESCARTES Dual Factory v3.0 -- Phase 1 Circuit 4 (Human MTL->Frontal)")
    logger.info("Data dir: %s", C4_DIR)

    # Load manifest
    manifest_path = C4_DIR / 'manifest.json'
    if not manifest_path.exists():
        logger.error("manifest.json not found in %s -- run extraction first", C4_DIR)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    logger.info("Manifest: %d patients", manifest['n_patients'])

    # Select patients
    if args.patient:
        patient_ids = [args.patient]
    else:
        patient_ids = sorted(manifest['patients'].keys())

    all_results = {}
    t_total = time.time()

    for i, pid in enumerate(patient_ids):
        patient_dir = C4_DIR / pid
        if not patient_dir.exists():
            logger.info("[%d/%d] SKIP %s: directory not found", i + 1, len(patient_ids), pid)
            continue

        logger.info("=" * 70)
        logger.info("[%d/%d] %s", i + 1, len(patient_ids), pid)
        logger.info("=" * 70)

        data = load_patient(patient_dir)
        cc = data['meta'].get('cc', 0)
        n_bin = data['h_bin_trained'].shape[0]
        n_win = data['h_win_trained'].shape[0]
        logger.info("  CC=%.3f  bin=%d  win=%d  targets=%s",
                     cc or 0, n_bin, n_win, data['target_names'])

        patient_results = {'meta': data['meta']}

        # Analysis 1: Hardening
        if args.analysis in (0, 1):
            logger.info("  --- Hardening ---")
            hardening = run_hardening_patient(data)
            patient_results['hardening'] = hardening
            print_hardening_table(pid, hardening, 'bin_level')
            print_hardening_table(pid, hardening, 'window_level')

        # Analysis 2: SAE
        if args.analysis in (0, 2):
            logger.info("  --- SAE ---")
            sae = run_sae_patient(data)
            patient_results['sae'] = sae

        # Analysis 3: MLP
        if args.analysis in (0, 3):
            logger.info("  --- MLP ---")
            mlp = run_mlp_patient(data)
            patient_results['mlp'] = mlp

        all_results[pid] = patient_results

        # Save per-patient result
        out_path = RESULTS_DIR / f'phase1_{pid}.json'
        with open(out_path, 'w') as f:
            json.dump(to_serializable(patient_results), f, indent=2)
        logger.info("  Saved: %s", out_path)

    total_time = time.time() - t_total
    logger.info("All patients complete in %.1f min", total_time / 60)

    # Cross-patient summary
    print(f"\n{'='*100}")
    print("CIRCUIT 4 CROSS-PATIENT SUMMARY")
    print(f"{'='*100}")

    for pid in sorted(all_results.keys()):
        pr = all_results[pid]
        cc = pr['meta'].get('cc', 0)
        print(f"\n  {pid} (CC={cc:.3f}):")

        if 'hardening' in pr:
            for level in ['bin_level', 'window_level']:
                verdicts = {}
                for name, r in pr['hardening'].get(level, {}).items():
                    if not r.get('skipped'):
                        v = r['hardened_verdict']
                        verdicts[v] = verdicts.get(v, 0) + 1
                if verdicts:
                    summary = ', '.join(f"{v}:{c}" for v, c in sorted(verdicts.items()))
                    print(f"    {level}: {summary}")

        if 'mlp' in pr:
            encoded = sum(1 for r in pr['mlp'].values()
                          if isinstance(r, dict) and
                          r.get('encoding_type', '').endswith('ENCODED'))
            total = len(pr['mlp'])
            print(f"    MLP: {encoded}/{total} encoded")

    # Save cross-patient summary
    summary_path = RESULTS_DIR / 'phase1_circuit4_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)
    logger.info("Summary saved: %s", summary_path)


if __name__ == '__main__':
    main()
