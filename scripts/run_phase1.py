#!/usr/bin/env python3
"""Phase 1 Orchestrator: Bahl reduced model DESCARTES zombie test.

Steps:
  1. Simulate 500 trials (Bahl 6-compartment L5PC)
  2. Train LSTM surrogates (h=64, 128, 256)
  3. Extract hidden states (trained + untrained baselines)
  4. Run 3-level Ridge probing (A: gates, B: G_eff, C: emergent)
  5. Run voltage-only baselines (Level B control)
  6. Run causal ablation on non-zombie variables
  7. Classify all variables (zombie / voltage re-encoding / byproduct / mandatory)
  8. Generate tables and figures

Usage:
    python scripts/run_phase1.py                    # Full pipeline
    python scripts/run_phase1.py --start-step 3     # Resume from step 3
    python scripts/run_phase1.py --only-step 4      # Run only step 4
"""
import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from l5pc.config import (
    BAHL_TRIAL_DIR, SURROGATE_DIR, RESULTS_DIR,
    HIDDEN_SIZES, N_TRIALS, DATA_DIR
)


def step_complete(step_name, output_dir):
    """Check if a step's outputs already exist."""
    checks = {
        'simulate': BAHL_TRIAL_DIR / 'trial_000.npz',
        'train': SURROGATE_DIR / f'lstm_h{HIDDEN_SIZES[-1]}_best.pt',
        'extract': SURROGATE_DIR / f'hidden_h{HIDDEN_SIZES[0]}_trained.npy',
        'probe': RESULTS_DIR / f'ridge_levelB_h{HIDDEN_SIZES[0]}.json',
        'baselines': RESULTS_DIR / 'voltage_baselines.json',
        'ablation': RESULTS_DIR / 'ablation_results.json',
        'classify': RESULTS_DIR / 'classification_summary.json',
        'visualize': RESULTS_DIR / 'figures' / f'delta_r2_levelB_h{HIDDEN_SIZES[0]}.png',
    }
    marker = checks.get(step_name)
    if marker and marker.exists():
        return True
    return False


def run_step_1_simulate(force=False):
    """Step 1: Run 500 Bahl simulations."""
    print("\n" + "=" * 70)
    print("STEP 1: Simulating Bahl L5PC trials")
    print("=" * 70)

    if not force and step_complete('simulate', BAHL_TRIAL_DIR):
        print("  SKIP: Simulation data already exists.")
        return

    from l5pc.simulation.run_bahl_sim import run_all_trials
    run_all_trials(
        n_trials=N_TRIALS,
        output_dir=str(BAHL_TRIAL_DIR),
        seed=42
    )
    print("  DONE: Simulation complete.")


def run_step_2_train(force=False):
    """Step 2: Train LSTM surrogates at 3 hidden sizes."""
    print("\n" + "=" * 70)
    print("STEP 2: Training LSTM surrogates")
    print("=" * 70)

    if not force and step_complete('train', SURROGATE_DIR):
        print("  SKIP: Trained models already exist.")
        return

    from l5pc.surrogates.train import train_lstm
    for h in HIDDEN_SIZES:
        model_path = SURROGATE_DIR / f'lstm_h{h}_best.pt'
        if model_path.exists() and not force:
            print(f"  SKIP h={h}: Model exists at {model_path}")
            continue
        print(f"\n  Training LSTM h={h}...")
        train_lstm(
            data_dir=str(BAHL_TRIAL_DIR),
            hidden_size=h,
            save_path=str(model_path)
        )
    print("  DONE: All LSTM models trained.")


def run_step_3_extract(force=False):
    """Step 3: Extract hidden states from trained + untrained models."""
    print("\n" + "=" * 70)
    print("STEP 3: Extracting hidden states")
    print("=" * 70)

    if not force and step_complete('extract', SURROGATE_DIR):
        print("  SKIP: Hidden states already extracted.")
        return

    from l5pc.surrogates.extract_hidden import extract_all
    extract_all(
        trial_dir=str(BAHL_TRIAL_DIR),
        model_dir=str(SURROGATE_DIR),
        save_dir=str(SURROGATE_DIR)
    )
    print("  DONE: Hidden states extracted.")


def run_step_4_probe(force=False):
    """Step 4: Run 3-level Ridge probing."""
    print("\n" + "=" * 70)
    print("STEP 4: Running 3-level Ridge probing")
    print("=" * 70)

    if not force and step_complete('probe', RESULTS_DIR):
        print("  SKIP: Probing results already exist.")
        return

    from l5pc.probing.ridge_probe import run_all_probes
    run_all_probes(
        hidden_dir=str(SURROGATE_DIR),
        targets_dir=str(BAHL_TRIAL_DIR),
        results_dir=str(RESULTS_DIR)
    )
    print("  DONE: Ridge probing complete.")


def run_step_5_baselines(force=False):
    """Step 5: Run voltage-only baselines for Level B."""
    print("\n" + "=" * 70)
    print("STEP 5: Computing voltage-only baselines")
    print("=" * 70)

    if not force and step_complete('baselines', RESULTS_DIR):
        print("  SKIP: Baseline results already exist.")
        return

    from l5pc.probing.baselines import run_voltage_baselines
    # Use h=128 as the primary hidden size for baseline comparison
    ridge_path = RESULTS_DIR / f'ridge_levelB_h128.json'
    run_voltage_baselines(
        trial_dir=str(BAHL_TRIAL_DIR),
        ridge_results_path=str(ridge_path),
        save_path=str(RESULTS_DIR / 'voltage_baselines.json')
    )
    print("  DONE: Voltage baselines computed.")


def run_step_6_ablation(force=False):
    """Step 6: Causal ablation on non-zombie variables."""
    print("\n" + "=" * 70)
    print("STEP 6: Running causal ablation")
    print("=" * 70)

    if not force and step_complete('ablation', RESULTS_DIR):
        print("  SKIP: Ablation results already exist.")
        return

    from l5pc.probing.ablation import run_all_ablations
    run_all_ablations(
        model_dir=str(SURROGATE_DIR),
        trial_dir=str(BAHL_TRIAL_DIR),
        ridge_dir=str(RESULTS_DIR),
        hidden_dir=str(SURROGATE_DIR),
        save_path=str(RESULTS_DIR / 'ablation_results.json')
    )
    print("  DONE: Causal ablation complete.")


def run_step_7_classify(force=False):
    """Step 7: Final classification of all variables."""
    print("\n" + "=" * 70)
    print("STEP 7: Classifying variables")
    print("=" * 70)

    if not force and step_complete('classify', RESULTS_DIR):
        print("  SKIP: Classification results already exist.")
        return

    from l5pc.probing.classify import classify_all, print_classification_summary
    results = classify_all(
        ridge_dir=str(RESULTS_DIR),
        baseline_path=str(RESULTS_DIR / 'voltage_baselines.json'),
        ablation_path=str(RESULTS_DIR / 'ablation_results.json'),
        save_path=str(RESULTS_DIR / 'classification_summary.json')
    )
    print_classification_summary(results)
    print("  DONE: Classification complete.")


def run_step_8_visualize(force=False):
    """Step 8: Generate all tables and figures."""
    print("\n" + "=" * 70)
    print("STEP 8: Generating visualization")
    print("=" * 70)

    fig_dir = RESULTS_DIR / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    from l5pc.visualization.probe_tables import print_all_tables, plot_all_levels, plot_cross_hidden_comparison
    from l5pc.visualization.ablation_curves import plot_all_ablations, plot_mandatory_summary

    # Tables
    print("\n  --- Ridge DR2 Tables ---")
    for h in HIDDEN_SIZES:
        print_all_tables(str(RESULTS_DIR), hidden_size=h)

    # Bar charts
    for h in HIDDEN_SIZES:
        plot_all_levels(str(RESULTS_DIR), hidden_size=h, save_dir=str(fig_dir))

    # Cross-hidden comparison
    for level in ['A', 'B', 'C']:
        plot_cross_hidden_comparison(
            str(RESULTS_DIR), level=level,
            save_path=str(fig_dir / f'cross_hidden_level{level}.png')
        )

    # Ablation curves
    ablation_path = RESULTS_DIR / 'ablation_results.json'
    if ablation_path.exists():
        plot_all_ablations(str(ablation_path), save_dir=str(fig_dir))
        plot_mandatory_summary(str(ablation_path), save_path=str(fig_dir / 'mandatory_summary.png'))

    print(f"  DONE: Figures saved to {fig_dir}")


STEPS = {
    1: ('simulate',  run_step_1_simulate),
    2: ('train',     run_step_2_train),
    3: ('extract',   run_step_3_extract),
    4: ('probe',     run_step_4_probe),
    5: ('baselines', run_step_5_baselines),
    6: ('ablation',  run_step_6_ablation),
    7: ('classify',  run_step_7_classify),
    8: ('visualize', run_step_8_visualize),
}


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Bahl L5PC DESCARTES zombie test')
    parser.add_argument('--start-step', type=int, default=1,
                        help='Start from this step (1-8)')
    parser.add_argument('--only-step', type=int, default=None,
                        help='Run only this step')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if outputs exist')
    args = parser.parse_args()

    print("=" * 70)
    print("L5PC DESCARTES - Phase 1: Bahl Reduced Model")
    print("=" * 70)
    t0 = time.time()

    if args.only_step:
        name, func = STEPS[args.only_step]
        func(force=args.force)
    else:
        for step_num in range(args.start_step, 9):
            name, func = STEPS[step_num]
            func(force=args.force)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Phase 1 complete in {elapsed / 60:.1f} minutes")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
