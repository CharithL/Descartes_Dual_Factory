#!/usr/bin/env python3
"""
DESCARTES Dual Factory v3.0 — Phase 1 Analysis: Hippocampal MIMO Circuit

Runs the same three analyses as L5PC Phase 1, on the hippocampal gamma oscillation
surrogate (La Masson 2002 MIMO model). This is the CONTROL circuit — gamma_amp
already has Delta-R2 = +0.177 at h=128 and survived resample ablation. If it also
survives statistical hardening + SAE + MLP, the cross-circuit dissociation
(L5PC zombie vs hippocampal non-zombie) is publishable.

Analyses:
  1. hardened_probe() on all 25 biological variables (h=128 = 256 dims, 2-layer LSTM)
  2. sae_probe_biological_variables() on h=128 (256d) and h=256 (512d)
  3. mlp_delta_r2() on all 25 variables (h=128)

Data layout (hippocampal_mimo/):
  checkpoints_rates/hidden_states/lstm_{trained,untrained}.npy  -> (21600, 256)
  checkpoints_rates/probe_targets.npy                           -> (21600, 25)
  checkpoints_rates/probe_variable_names.json                   -> 25 names
  checkpoints_rates/test_data.npz  X=(108,200,14)               -> inputs
  sweep_h256/{trained,untrained}_hidden.npy                     -> (21600, 512)

Usage:
    python scripts/run_phase1_hippocampal.py [--analysis 0|1|2|3]
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path (Descartes_Dual_Factory)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Configuration ──────────────────────────────────────────────────────
# HIPPO_ROOT must point to the hippocampal_mimo directory.
# On Vast.ai this will be unpacked from the data zip.
HIPPO_ROOT = Path(__file__).parent.parent / 'hippocampal_mimo'

# If running from inside the hippocampal_mimo directory itself:
if not HIPPO_ROOT.exists():
    HIPPO_ROOT = PROJECT_ROOT

RATES_DIR = HIPPO_ROOT / 'checkpoints_rates'
HIDDEN_DIR = RATES_DIR / 'hidden_states'
RESULTS_DIR = HIPPO_ROOT / 'results_v3'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hippocampal LSTM: 2 layers, so hidden dim = 2 * nominal_h
# "h=128" checkpoint has 256 dims; "h=256" has 512 dims
T_STEPS = 200       # timesteps per trial at 25ms resolution
N_TEST_TRIALS = 108  # test split
SAMPLING_RATE = 40   # 1000ms / 25ms = 40 Hz

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / 'phase1_hippocampal.log'),
    ],
)
logger = logging.getLogger('phase1_hippo')


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_hidden_states_hippo(nominal_h, trained=True):
    """Load hippocampal LSTM hidden states.

    Parameters
    ----------
    nominal_h : int
        Nominal hidden size (128 or 256). Actual dims = 2 * nominal_h
        because the LSTM has 2 layers and hidden states are concatenated.
    trained : bool
        Whether to load trained or untrained checkpoint.

    Returns
    -------
    ndarray (21600, 2*nominal_h)
    """
    tag = 'trained' if trained else 'untrained'

    if nominal_h == 128:
        # Default checkpoint in checkpoints_rates/hidden_states/
        path = HIDDEN_DIR / f'lstm_{tag}.npy'
    else:
        # Sweep checkpoints
        path = HIPPO_ROOT / f'sweep_h{nominal_h}' / f'{tag}_hidden.npy'

    h = np.load(path)
    logger.info("Loaded %s h=%d: shape %s  (path: %s)", tag, nominal_h, h.shape, path.name)
    return h


def load_probe_targets():
    """Load pre-prepared probe targets (21600, 25) and variable names."""
    targets = np.load(RATES_DIR / 'probe_targets.npy')
    with open(RATES_DIR / 'probe_variable_names.json') as f:
        var_names = json.load(f)
    logger.info("Loaded probe targets: shape %s, %d variables", targets.shape, len(var_names))
    return targets, var_names


def load_inputs():
    """Load input signals for test split (108, 200, 14) -> (21600, 14)."""
    td = np.load(RATES_DIR / 'test_data.npz')
    X = td['X']  # (108, 200, 14)
    inputs = X.reshape(-1, X.shape[-1])  # (21600, 14)
    logger.info("Loaded inputs: shape %s", inputs.shape)
    return inputs


# ═══════════════════════════════════════════════════════════════════════
# JSON serialization helper
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# Analysis 1: Statistical Hardening
# ═══════════════════════════════════════════════════════════════════════

def run_hardening_analysis(nominal_h=128):
    """Run hardened_probe() on all 25 biological variables."""
    from l5pc.probing.hardening import hardened_probe

    logger.info("=" * 70)
    logger.info("ANALYSIS 1: Statistical Hardening — Hippocampal h=%d (%d dims)",
                nominal_h, nominal_h * 2)
    logger.info("=" * 70)

    h_trained = load_hidden_states_hippo(nominal_h, trained=True)
    h_untrained = load_hidden_states_hippo(nominal_h, trained=False)
    targets, var_names = load_probe_targets()
    inputs = load_inputs()

    # No subsampling needed: only 21,600 samples (vs L5PC's 150,000)
    logger.info("Samples: %d (no subsampling needed)", h_trained.shape[0])

    device = 'cpu'
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
        target_col = targets[:, j].astype(np.float64)

        if target_col.std() < 1e-10:
            logger.info("[%2d/%d] %s: SKIPPED (zero variance)", j+1, len(var_names), name)
            results[name] = {'skipped': True, 'reason': 'zero_variance'}
            continue

        logger.info("[%2d/%d] %s ...", j+1, len(var_names), name)
        t1 = time.time()

        result = hardened_probe(
            hidden_trained=h_trained.astype(np.float64),
            hidden_untrained=h_untrained.astype(np.float64),
            target=target_col,
            target_name=name,
            input_signal=inputs.astype(np.float64),
            sampling_rate=SAMPLING_RATE,
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

    out_path = RESULTS_DIR / f'hardened_hippo_h{nominal_h}.json'
    with open(out_path, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    logger.info("Saved: %s", out_path)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"HARDENING RESULTS — Hippocampal, h={nominal_h} ({nominal_h*2} dims)")
    print("=" * 90)
    print(f"{'Variable':<25} {'Verdict':<30} {'Ridge dR2':>10} {'MLP dR2':>10} {'p_block':>8} {'BF01':>8}")
    print("-" * 90)
    for name in var_names:
        r = results[name]
        if r.get('skipped'):
            print(f"{name:<25} {'SKIPPED':<30}")
            continue
        print(f"{name:<25} {r['hardened_verdict']:<30} "
              f"{r['ridge_delta_r2']:>10.4f} {r['mlp_delta_r2']:>10.4f} "
              f"{r['p_block_permutation']:>8.4f} {r['bayes_factor']['bf01']:>8.2f}")
    print("=" * 90)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Analysis 2: SAE Superposition Detection
# ═══════════════════════════════════════════════════════════════════════

def run_sae_analysis(hidden_sizes=(128, 256)):
    """Run SAE decomposition + probing on hippocampal hidden states."""
    from l5pc.probing.sae_probe import train_sae, sae_probe_biological_variables

    logger.info("=" * 70)
    logger.info("ANALYSIS 2: SAE Superposition Detection — Hippocampal")
    logger.info("=" * 70)

    targets, var_names = load_probe_targets()

    device = 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info("Using CUDA for SAE training")
    except ImportError:
        pass

    all_results = {}

    for nominal_h in hidden_sizes:
        actual_dim = nominal_h * 2  # 2-layer LSTM concatenated
        logger.info("\n--- SAE for h=%d (%d dims) ---", nominal_h, actual_dim)
        h_trained = load_hidden_states_hippo(nominal_h, trained=True)

        # Split into trial-shaped lists for train_sae
        hidden_list = [
            h_trained[i * T_STEPS:(i + 1) * T_STEPS]
            for i in range(N_TEST_TRIALS)
        ]
        bio_list = [
            targets[i * T_STEPS:(i + 1) * T_STEPS]
            for i in range(N_TEST_TRIALS)
        ]

        for exp_factor in [4, 8]:
            logger.info("Training SAE: expansion=%dx, k=20, epochs=50", exp_factor)
            t0 = time.time()

            sae, loss_history = train_sae(
                hidden_list, actual_dim,
                expansion_factor=exp_factor, k=20,
                epochs=50, batch_size=4096, device=device,
            )
            train_time = time.time() - t0
            logger.info("SAE trained in %.1f min, final loss=%.6f",
                        train_time / 60, loss_history[-1])

            logger.info("Probing SAE features for %d biological variables...", len(var_names))
            t0 = time.time()
            probe_result = sae_probe_biological_variables(
                sae, hidden_list, bio_list, var_names, device=device,
            )
            probe_time = time.time() - t0
            logger.info("Probing complete in %.1f min", probe_time / 60)

            key = f'h{nominal_h}_exp{exp_factor}'
            all_results[key] = {
                'nominal_h': nominal_h,
                'actual_dim': actual_dim,
                'expansion_factor': exp_factor,
                'loss_history': [float(l) for l in loss_history],
                'n_alive': probe_result['n_alive'],
                'mean_monosemanticity': probe_result['mean_monosemanticity'],
                'sae_r2': {k: float(v) for k, v in probe_result['sae_r2'].items()},
                'raw_r2': {k: float(v) for k, v in probe_result['raw_r2'].items()},
                'superposition_detected': probe_result['superposition_detected'],
            }

            # Print table
            total_feats = exp_factor * actual_dim
            print(f"\n{'='*80}")
            print(f"SAE RESULTS — Hippocampal h={nominal_h} ({actual_dim}d), expansion={exp_factor}x")
            print(f"Alive features: {probe_result['n_alive']}/{total_feats}")
            print(f"Mean monosemanticity: {probe_result['mean_monosemanticity']:.4f}")
            print(f"{'='*80}")
            print(f"{'Variable':<25} {'Raw R2':>8} {'SAE R2':>8} {'Gain':>8} {'Superposed?':>12}")
            print("-" * 80)

            for name in var_names:
                raw = probe_result['raw_r2'][name]
                sae_r = probe_result['sae_r2'][name]
                gain = sae_r - raw
                sup = probe_result['superposition_detected'][name]
                marker = " *** SUPERPOSED ***" if sup else ""
                print(f"{name:<25} {raw:>8.4f} {sae_r:>8.4f} {gain:>8.4f} "
                      f"{'YES' if sup else 'no':>12}{marker}")
            print("=" * 80)

    out_path = RESULTS_DIR / 'sae_superposition_hippo.json'
    with open(out_path, 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)
    logger.info("Saved: %s", out_path)

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Analysis 3: MLP Nonlinear Encoding Check
# ═══════════════════════════════════════════════════════════════════════

def run_mlp_analysis(nominal_h=128):
    """Run MLP delta-R2 on all 25 hippocampal variables."""
    from l5pc.probing.mlp_probe import mlp_delta_r2

    logger.info("=" * 70)
    logger.info("ANALYSIS 3: MLP Nonlinear Encoding Check — Hippocampal h=%d", nominal_h)
    logger.info("=" * 70)

    h_trained = load_hidden_states_hippo(nominal_h, trained=True)
    h_untrained = load_hidden_states_hippo(nominal_h, trained=False)
    targets, var_names = load_probe_targets()

    device = 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info("Using CUDA for MLP probing")
    except ImportError:
        pass

    # No subsampling — 21,600 samples is manageable
    logger.info("Running MLP delta-R2 on %d variables (epochs=50, hidden=64)...", len(var_names))

    t0 = time.time()
    results = mlp_delta_r2(
        h_trained.astype(np.float32),
        h_untrained.astype(np.float32),
        targets.astype(np.float32),
        var_names, hidden_dim=64, epochs=50, lr=1e-3,
        n_splits=5, device=device,
    )
    elapsed = time.time() - t0
    logger.info("MLP analysis complete in %.1f min", elapsed / 60)

    out_path = RESULTS_DIR / f'mlp_delta_r2_hippo_h{nominal_h}.json'
    with open(out_path, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    logger.info("Saved: %s", out_path)

    # Print table
    print(f"\n{'='*100}")
    print(f"MLP vs RIDGE — Hippocampal, h={nominal_h} ({nominal_h*2} dims)")
    print(f"{'='*100}")
    print(f"{'Variable':<25} {'Ridge dR2':>10} {'MLP dR2':>10} {'Gain':>8} {'Encoding Type':<22} {'Verdict'}")
    print("-" * 100)

    non_zombie = 0
    for name in var_names:
        r = results[name]
        ridge_d = r['ridge_delta']
        mlp_d = r['mlp_delta']
        gain = r['nonlinear_gain']
        enc = r['encoding_type']

        is_encoded = enc in ('LINEAR_ENCODED', 'NONLINEAR_ENCODED', 'NONLINEAR_ONLY')
        marker = " <<< ENCODED" if is_encoded else ""
        if is_encoded:
            non_zombie += 1

        print(f"{name:<25} {ridge_d:>10.4f} {mlp_d:>10.4f} {gain:>8.4f} {enc:<22}{marker}")

    print("=" * 100)
    print(f"\nEncoded variables: {non_zombie}/{len(var_names)}")
    print(f"Zombie variables: {len(var_names) - non_zombie}/{len(var_names)}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DESCARTES Phase 1 — Hippocampal MIMO')
    parser.add_argument('--analysis', type=int, default=0,
                        help='Run specific analysis (1=hardening, 2=SAE, 3=MLP, 0=all)')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Nominal hidden size for hardening/MLP (default: 128)')
    args = parser.parse_args()

    logger.info("DESCARTES Dual Factory v3.0 — Phase 1 Hippocampal Analysis")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Hippocampal data: %s", HIPPO_ROOT)

    if args.analysis == 0 or args.analysis == 1:
        run_hardening_analysis(args.hidden_size)

    if args.analysis == 0 or args.analysis == 2:
        run_sae_analysis(hidden_sizes=(128, 256))

    if args.analysis == 0 or args.analysis == 3:
        run_mlp_analysis(args.hidden_size)

    logger.info("All analyses complete. Results saved to %s", RESULTS_DIR)
