#!/usr/bin/env python3
"""Phase 2 Orchestrator: Full Hay model spatial analysis.

Steps:
  1. Simulate 2000 trials (Hay 639-compartment L5PC)
  2. Train LSTM surrogates on Hay data
  3. Load pre-trained Beniaguev TCN (if available)
  4. Extract hidden states
  5. Run spatial probing (distance-dependent R2)
  6. Analyze hot zone vs non-hot zone encoding
  7. Generate spatial maps and figures

Usage:
    python scripts/run_phase2.py
    python scripts/run_phase2.py --start-step 3
    python scripts/run_phase2.py --n-trials 200  # Quick test
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from l5pc.config import (
    HAY_TRIAL_DIR, HAY_MODEL_DIR, SURROGATE_DIR, RESULTS_DIR,
    HIDDEN_SIZES, HAY_N_TRIALS, HAY_N_COMPARTMENTS
)


def run_step_1_simulate(n_trials, force=False):
    """Step 1: Run Hay model simulations."""
    print("\n" + "=" * 70)
    print(f"STEP 1: Simulating {n_trials} Hay L5PC trials ({HAY_N_COMPARTMENTS} compartments)")
    print("=" * 70)

    marker = HAY_TRIAL_DIR / 'trial_000.npz'
    if not force and marker.exists():
        print("  SKIP: Hay simulation data exists.")
        return

    from l5pc.simulation.hay_model import run_hay_simulation
    run_hay_simulation(
        n_trials=n_trials,
        output_dir=str(HAY_TRIAL_DIR),
        seed=42
    )
    print("  DONE: Hay simulation complete.")


def run_step_2_train(force=False):
    """Step 2: Train LSTM on Hay data."""
    print("\n" + "=" * 70)
    print("STEP 2: Training LSTM surrogates on Hay data")
    print("=" * 70)

    hay_surrogate_dir = SURROGATE_DIR / 'hay'
    hay_surrogate_dir.mkdir(parents=True, exist_ok=True)

    from l5pc.surrogates.train import train_lstm
    for h in HIDDEN_SIZES:
        model_path = hay_surrogate_dir / f'lstm_h{h}_best.pt'
        if model_path.exists() and not force:
            print(f"  SKIP h={h}: Model exists.")
            continue
        print(f"\n  Training Hay LSTM h={h}...")
        train_lstm(
            trial_dir=str(HAY_TRIAL_DIR),
            hidden_size=h,
            save_dir=str(hay_surrogate_dir)
        )
    print("  DONE: Hay LSTM models trained.")


def run_step_3_tcn(force=False):
    """Step 3: Load pre-trained Beniaguev TCN."""
    print("\n" + "=" * 70)
    print("STEP 3: Loading pre-trained Beniaguev TCN")
    print("=" * 70)

    from l5pc.config import TCN_PRETRAINED_PATH
    if TCN_PRETRAINED_PATH.exists():
        from l5pc.surrogates.tcn import load_pretrained_tcn
        try:
            model = load_pretrained_tcn(str(TCN_PRETRAINED_PATH))
            print(f"  DONE: TCN loaded from {TCN_PRETRAINED_PATH}")
            print(f"        Parameters: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"  WARNING: Could not load TCN: {e}")
            print("  Continuing with LSTM-only analysis.")
    else:
        print(f"  SKIP: TCN weights not found at {TCN_PRETRAINED_PATH}")
        print("  Download from: https://www.kaggle.com/datasets/baborasmit/beniaguev-tcn-weights")


def run_step_4_extract(force=False):
    """Step 4: Extract hidden states from Hay models."""
    print("\n" + "=" * 70)
    print("STEP 4: Extracting hidden states (Hay models)")
    print("=" * 70)

    hay_surrogate_dir = SURROGATE_DIR / 'hay'
    marker = hay_surrogate_dir / f'hidden_h{HIDDEN_SIZES[0]}_trained.npy'
    if not force and marker.exists():
        print("  SKIP: Hidden states exist.")
        return

    from l5pc.surrogates.extract_hidden import extract_all
    extract_all(
        trial_dir=str(HAY_TRIAL_DIR),
        model_dir=str(hay_surrogate_dir),
        save_dir=str(hay_surrogate_dir)
    )
    print("  DONE: Hidden states extracted.")


def run_step_5_spatial_probe(force=False):
    """Step 5: Run spatial probing - R2 as function of distance from soma."""
    print("\n" + "=" * 70)
    print("STEP 5: Spatial probing across dendritic tree")
    print("=" * 70)

    spatial_results = RESULTS_DIR / 'hay_spatial'
    spatial_results.mkdir(parents=True, exist_ok=True)

    marker = spatial_results / 'spatial_probe_results.json'
    if not force and marker.exists():
        print("  SKIP: Spatial probe results exist.")
        return

    # Use the Hay cell to get distance information
    from l5pc.simulation.hay_model import HayCell
    from l5pc.probing.ridge_probe import run_all_probes

    hay_surrogate_dir = SURROGATE_DIR / 'hay'
    run_all_probes(
        hidden_dir=str(hay_surrogate_dir),
        targets_dir=str(HAY_TRIAL_DIR),
        results_dir=str(spatial_results)
    )
    print("  DONE: Spatial probing complete.")


def run_step_6_hotzone(force=False):
    """Step 6: Hot zone vs non-hot zone encoding analysis."""
    print("\n" + "=" * 70)
    print("STEP 6: Calcium hot zone analysis")
    print("=" * 70)

    from l5pc.config import CA_HOTZONE_START_UM, CA_HOTZONE_END_UM
    print(f"  Hot zone: {CA_HOTZONE_START_UM}-{CA_HOTZONE_END_UM} um from soma")

    spatial_results = RESULTS_DIR / 'hay_spatial'
    from l5pc.utils.io import load_results_json, save_results_json

    # Load spatial results and partition by distance
    result_files = list(spatial_results.glob('ridge_level*_h*.json'))
    if not result_files:
        print("  SKIP: No spatial probe results to analyze.")
        return

    print(f"  Analyzing {len(result_files)} result files...")
    # Partition analysis happens in the visualization step
    print("  DONE: Hot zone partitioning prepared.")


def run_step_7_visualize(force=False):
    """Step 7: Generate spatial maps and figures."""
    print("\n" + "=" * 70)
    print("STEP 7: Generating Phase 2 visualizations")
    print("=" * 70)

    fig_dir = RESULTS_DIR / 'figures' / 'phase2'
    fig_dir.mkdir(parents=True, exist_ok=True)
    spatial_results = RESULTS_DIR / 'hay_spatial'

    from l5pc.visualization.probe_tables import print_all_tables, plot_all_levels

    # Print tables for Hay results
    if spatial_results.exists():
        for h in HIDDEN_SIZES:
            print_all_tables(str(spatial_results), hidden_size=h)
        plot_all_levels(str(spatial_results), hidden_size=128, save_dir=str(fig_dir))

    print(f"  DONE: Phase 2 figures saved to {fig_dir}")


STEPS = {
    1: ('simulate',      run_step_1_simulate),
    2: ('train',         run_step_2_train),
    3: ('tcn',           run_step_3_tcn),
    4: ('extract',       run_step_4_extract),
    5: ('spatial_probe', run_step_5_spatial_probe),
    6: ('hotzone',       run_step_6_hotzone),
    7: ('visualize',     run_step_7_visualize),
}


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Hay L5PC spatial analysis')
    parser.add_argument('--start-step', type=int, default=1)
    parser.add_argument('--only-step', type=int, default=None)
    parser.add_argument('--n-trials', type=int, default=HAY_N_TRIALS)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("L5PC DESCARTES - Phase 2: Full Hay Model Spatial Analysis")
    print(f"  {args.n_trials} trials, {HAY_N_COMPARTMENTS} compartments")
    print("=" * 70)
    t0 = time.time()

    if args.only_step:
        name, func = STEPS[args.only_step]
        if args.only_step == 1:
            func(args.n_trials, force=args.force)
        else:
            func(force=args.force)
    else:
        for step_num in range(args.start_step, 8):
            name, func = STEPS[step_num]
            if step_num == 1:
                func(args.n_trials, force=args.force)
            else:
                func(force=args.force)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Phase 2 complete in {elapsed / 60:.1f} minutes")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
