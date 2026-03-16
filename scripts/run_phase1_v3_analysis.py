#!/usr/bin/env python3
"""
DESCARTES Dual Factory v3.0 — Phase 1 Analysis on Existing Checkpoints

Runs three critical analyses on Bahl model LSTM surrogates:

  1. hardened_probe() on Level B borderline zombies (h=128)
     → Formal significance testing for every delta-R2

  2. sae_probe_biological_variables() on h=128 and h=256
     → Superposition test: is biology encoded but entangled?

  3. mlp_delta_r2() on Level B zombies (h=128)
     → Nonlinear encoding check: does MLP catch what Ridge misses?

Usage:
    python scripts/run_phase1_v3_analysis.py

Expects:
    data/surrogates/hidden/lstm_{128,256}_{trained,untrained}.npz
    data/bahl_trials/trial_*.npz (test split: trials 425-499)
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from l5pc.config import TRAIN_SPLIT, VAL_SPLIT, N_TRIALS, T_STEPS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'data' / 'results' / 'phase1_v3_analysis.log'),
    ],
)
logger = logging.getLogger('phase1_v3')

DATA_DIR = PROJECT_ROOT / 'data'
TRIAL_DIR = DATA_DIR / 'bahl_trials'
HIDDEN_DIR = DATA_DIR / 'surrogates' / 'hidden'
RESULTS_DIR = DATA_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_hidden_states(hidden_size, trained=True):
    """Load hidden states (n_test_trials * T_STEPS, hidden_dim)."""
    tag = 'trained' if trained else 'untrained'
    path = HIDDEN_DIR / f'lstm_{hidden_size}_{tag}.npz'
    data = np.load(path)
    h = data['hidden_states']
    logger.info("Loaded %s h=%d: shape %s", tag, hidden_size, h.shape)
    return h


def load_targets_level_b():
    """Load Level B effective conductance targets for test split.

    Returns (n_test_trials * T_STEPS, 50) array and list of 50 variable names.
    """
    with open(TRIAL_DIR / 'variable_names.json') as f:
        names = json.load(f)
    var_names = names['level_B']

    test_start = TRAIN_SPLIT + VAL_SPLIT  # 425
    test_end = N_TRIALS  # 500

    all_targets = []
    for i in range(test_start, test_end):
        trial = np.load(TRIAL_DIR / f'trial_{i:03d}.npz')
        all_targets.append(trial['level_B_cond'])  # (T_STEPS, 50)

    targets = np.concatenate(all_targets, axis=0)  # (75 * 2000, 50)
    logger.info("Loaded Level B targets: shape %s, %d variables", targets.shape, len(var_names))
    return targets, var_names


def load_inputs():
    """Load input signals for test split (for partial coherence)."""
    test_start = TRAIN_SPLIT + VAL_SPLIT
    test_end = N_TRIALS

    all_inputs = []
    for i in range(test_start, test_end):
        trial = np.load(TRIAL_DIR / f'trial_{i:03d}.npz')
        all_inputs.append(trial['inputs'])

    inputs = np.concatenate(all_inputs, axis=0)
    logger.info("Loaded inputs: shape %s", inputs.shape)
    return inputs


# ═══════════════════════════════════════════════════════════════════════
# Analysis 1: Statistical Hardening
# ═══════════════════════════════════════════════════════════════════════

def run_hardening_analysis(hidden_size=128):
    """Run hardened_probe() on all Level B variables for given hidden size.

    Focus on borderline zombies (Ridge dR2 between 0.05 and 0.2) but
    run ALL variables for complete table.
    """
    from l5pc.probing.hardening import hardened_probe

    logger.info("=" * 70)
    logger.info("ANALYSIS 1: Statistical Hardening (h=%d)", hidden_size)
    logger.info("=" * 70)

    h_trained = load_hidden_states(hidden_size, trained=True)
    h_untrained = load_hidden_states(hidden_size, trained=False)
    targets, var_names = load_targets_level_b()
    inputs = load_inputs()

    # Subsample for speed: use every 4th timestep (still 37,500 samples)
    # This is enough for statistical significance while cutting runtime 4x
    step = 4
    h_trained_sub = h_trained[::step]
    h_untrained_sub = h_untrained[::step]
    targets_sub = targets[::step]
    inputs_sub = inputs[::step]

    logger.info("Subsampled: %d -> %d samples (step=%d)",
                h_trained.shape[0], h_trained_sub.shape[0], step)

    device = 'cpu'  # MLP probe inside hardened_probe uses this

    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info("Using CUDA for MLP probing")
    except ImportError:
        pass

    results = {}
    t0 = time.time()

    for j, name in enumerate(var_names):
        target_col = targets_sub[:, j].astype(np.float64)

        # Skip if target has zero variance (constant)
        if target_col.std() < 1e-10:
            logger.info("[%2d/%d] %s: SKIPPED (zero variance)", j+1, len(var_names), name)
            results[name] = {'skipped': True, 'reason': 'zero_variance'}
            continue

        logger.info("[%2d/%d] %s ...", j+1, len(var_names), name)
        t1 = time.time()

        result = hardened_probe(
            hidden_trained=h_trained_sub.astype(np.float64),
            hidden_untrained=h_untrained_sub.astype(np.float64),
            target=target_col,
            target_name=name,
            input_signal=inputs_sub.astype(np.float64),
            sampling_rate=int(1000 / 0.5 / step),  # Adjusted for subsampling
            device=device,
        )
        elapsed = time.time() - t1

        results[name] = result
        logger.info(
            "  -> verdict=%s  ridge_dR2=%.4f  mlp_dR2=%.4f  "
            "p_block=%.4f  BF01=%.2f  DW=%.2f  [%.1fs]",
            result['hardened_verdict'],
            result['ridge_delta_r2'],
            result['mlp_delta_r2'],
            result['p_block_permutation'],
            result['bayes_factor']['bf01'],
            result['durbin_watson'],
            elapsed,
        )

    total_time = time.time() - t0
    logger.info("Hardening complete: %d variables in %.1f min", len(var_names), total_time / 60)

    # Save results
    # Convert numpy types for JSON serialization
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

    out_path = RESULTS_DIR / f'hardened_levelB_h{hidden_size}.json'
    with open(out_path, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    logger.info("Saved: %s", out_path)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"HARDENING RESULTS — Level B, h={hidden_size}")
    print("=" * 90)
    print(f"{'Variable':<30} {'Verdict':<28} {'Ridge dR2':>10} {'MLP dR2':>10} {'p_block':>8} {'BF01':>8}")
    print("-" * 90)
    for name in var_names:
        r = results[name]
        if r.get('skipped'):
            print(f"{name:<30} {'SKIPPED':<28} {'':>10} {'':>10} {'':>8} {'':>8}")
            continue
        print(f"{name:<30} {r['hardened_verdict']:<28} "
              f"{r['ridge_delta_r2']:>10.4f} {r['mlp_delta_r2']:>10.4f} "
              f"{r['p_block_permutation']:>8.4f} {r['bayes_factor']['bf01']:>8.2f}")
    print("=" * 90)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Analysis 2: SAE Superposition Detection
# ═══════════════════════════════════════════════════════════════════════

def run_sae_analysis(hidden_sizes=(128, 256)):
    """Run SAE decomposition + probing on Level B for h=128 and h=256."""
    from l5pc.probing.sae_probe import train_sae, sae_probe_biological_variables

    logger.info("=" * 70)
    logger.info("ANALYSIS 2: SAE Superposition Detection")
    logger.info("=" * 70)

    targets, var_names = load_targets_level_b()

    device = 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info("Using CUDA for SAE training")
    except ImportError:
        pass

    all_results = {}

    for hidden_size in hidden_sizes:
        logger.info("\n--- SAE for h=%d ---", hidden_size)
        h_trained = load_hidden_states(hidden_size, trained=True)

        # Split into trial-shaped lists for train_sae
        n_trials = h_trained.shape[0] // T_STEPS
        hidden_list = [
            h_trained[i * T_STEPS:(i + 1) * T_STEPS]
            for i in range(n_trials)
        ]
        bio_list = [
            targets[i * T_STEPS:(i + 1) * T_STEPS]
            for i in range(n_trials)
        ]

        # Train SAE at multiple expansion factors
        for exp_factor in [4, 8]:
            logger.info("Training SAE: expansion=%dx, k=20, epochs=50", exp_factor)
            t0 = time.time()

            sae, loss_history = train_sae(
                hidden_list, hidden_size,
                expansion_factor=exp_factor, k=20,
                epochs=50, batch_size=4096, device=device,
            )
            train_time = time.time() - t0
            logger.info("SAE trained in %.1f min, final loss=%.6f",
                        train_time / 60, loss_history[-1])

            # Probe
            logger.info("Probing SAE features for %d biological variables...", len(var_names))
            t0 = time.time()
            probe_result = sae_probe_biological_variables(
                sae, hidden_list, bio_list, var_names, device=device,
            )
            probe_time = time.time() - t0
            logger.info("Probing complete in %.1f min", probe_time / 60)

            key = f'h{hidden_size}_exp{exp_factor}'
            all_results[key] = {
                'hidden_size': hidden_size,
                'expansion_factor': exp_factor,
                'loss_history': [float(l) for l in loss_history],
                'n_alive': probe_result['n_alive'],
                'mean_monosemanticity': probe_result['mean_monosemanticity'],
                'sae_r2': {k: float(v) for k, v in probe_result['sae_r2'].items()},
                'raw_r2': {k: float(v) for k, v in probe_result['raw_r2'].items()},
                'superposition_detected': probe_result['superposition_detected'],
            }

            # Print per-variable comparison
            print(f"\n{'='*80}")
            print(f"SAE RESULTS — h={hidden_size}, expansion={exp_factor}x")
            print(f"Alive features: {probe_result['n_alive']}/{exp_factor * hidden_size}")
            print(f"Mean monosemanticity: {probe_result['mean_monosemanticity']:.4f}")
            print(f"{'='*80}")
            print(f"{'Variable':<30} {'Raw R2':>8} {'SAE R2':>8} {'Gain':>8} {'Superposed?':>12}")
            print("-" * 80)

            for name in var_names:
                raw = probe_result['raw_r2'][name]
                sae_r = probe_result['sae_r2'][name]
                gain = sae_r - raw
                sup = probe_result['superposition_detected'][name]
                marker = " *** SUPERPOSED ***" if sup else ""
                print(f"{name:<30} {raw:>8.4f} {sae_r:>8.4f} {gain:>8.4f} "
                      f"{'YES' if sup else 'no':>12}{marker}")
            print("=" * 80)

    # Save
    def to_serializable(obj):
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

    out_path = RESULTS_DIR / 'sae_superposition_levelB.json'
    with open(out_path, 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)
    logger.info("Saved: %s", out_path)

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Analysis 3: MLP Nonlinear Encoding Check on Zombies
# ═══════════════════════════════════════════════════════════════════════

def run_mlp_zombie_check(hidden_size=128):
    """Run MLP delta-R2 on all Level B variables (focus on zombies)."""
    from l5pc.probing.mlp_probe import mlp_delta_r2

    logger.info("=" * 70)
    logger.info("ANALYSIS 3: MLP Nonlinear Encoding Check (h=%d)", hidden_size)
    logger.info("=" * 70)

    h_trained = load_hidden_states(hidden_size, trained=True)
    h_untrained = load_hidden_states(hidden_size, trained=False)
    targets, var_names = load_targets_level_b()

    device = 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info("Using CUDA for MLP probing")
    except ImportError:
        pass

    # Subsample: every 4th timestep
    step = 4
    h_trained_sub = h_trained[::step].astype(np.float32)
    h_untrained_sub = h_untrained[::step].astype(np.float32)
    targets_sub = targets[::step].astype(np.float32)

    logger.info("Subsampled: %d -> %d samples", h_trained.shape[0], h_trained_sub.shape[0])
    logger.info("Running MLP delta-R2 on %d variables (epochs=50, hidden=64)...", len(var_names))

    t0 = time.time()
    results = mlp_delta_r2(
        h_trained_sub, h_untrained_sub, targets_sub,
        var_names, hidden_dim=64, epochs=50, lr=1e-3,
        n_splits=5, device=device,
    )
    elapsed = time.time() - t0
    logger.info("MLP analysis complete in %.1f min", elapsed / 60)

    # Save
    def to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        return obj

    out_path = RESULTS_DIR / f'mlp_delta_r2_levelB_h{hidden_size}.json'
    with open(out_path, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    logger.info("Saved: %s", out_path)

    # Print table
    print(f"\n{'='*100}")
    print(f"MLP vs RIDGE — Level B, h={hidden_size}")
    print(f"{'='*100}")
    print(f"{'Variable':<30} {'Ridge dR2':>10} {'MLP dR2':>10} {'Gain':>8} {'Encoding Type':<22} {'Verdict Change?'}")
    print("-" * 100)

    flips = 0
    for name in var_names:
        r = results[name]
        ridge_d = r['ridge_delta']
        mlp_d = r['mlp_delta']
        gain = r['nonlinear_gain']
        enc = r['encoding_type']

        # Was it a zombie by Ridge alone?
        was_zombie = ridge_d < 0.1
        now_rescued = was_zombie and enc in ('NONLINEAR_ENCODED', 'NONLINEAR_ONLY')
        marker = " <<< RESCUED FROM ZOMBIE" if now_rescued else ""
        if now_rescued:
            flips += 1

        print(f"{name:<30} {ridge_d:>10.4f} {mlp_d:>10.4f} {gain:>8.4f} {enc:<22}{marker}")

    print("=" * 100)
    print(f"\nVerdicts rescued from ZOMBIE by MLP: {flips}/{len(var_names)}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DESCARTES Phase 1 v3.0 Analysis')
    parser.add_argument('--analysis', type=int, default=0,
                        help='Run specific analysis (1=hardening, 2=SAE, 3=MLP, 0=all)')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Hidden size for hardening/MLP (default: 128)')
    args = parser.parse_args()

    logger.info("DESCARTES Dual Factory v3.0 — Phase 1 Analysis")
    logger.info("Project root: %s", PROJECT_ROOT)

    if args.analysis == 0 or args.analysis == 1:
        hardening_results = run_hardening_analysis(args.hidden_size)

    if args.analysis == 0 or args.analysis == 2:
        sae_results = run_sae_analysis(hidden_sizes=(128, 256))

    if args.analysis == 0 or args.analysis == 3:
        mlp_results = run_mlp_zombie_check(args.hidden_size)

    logger.info("All analyses complete. Results saved to %s", RESULTS_DIR)
