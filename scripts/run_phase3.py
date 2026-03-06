#!/usr/bin/env python3
"""Phase 3 Orchestrator: Circuit integration and graded replacement.

Steps:
  1. Build 100-neuron cortical microcircuit (NetPyNE)
  2. Run biophysical baseline simulations
  3. Graded surrogate replacement (0% -> single -> 10% -> 25% -> 50% -> 100%)
  4. Level 1 validation: Output fidelity
  5. Level 2 validation: Circuit-level dynamics
  6. Level 3 validation: DESCARTES mechanistic (re-probe in circuit context)
  7. Level 4 validation: Consciousness-relevant metrics
  8. Generate replacement curves and 4-level dashboard

Usage:
    python scripts/run_phase3.py
    python scripts/run_phase3.py --start-step 3
    python scripts/run_phase3.py --replicates 3  # Quick test
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from l5pc.config import (
    SURROGATE_DIR, RESULTS_DIR,
    REPLACEMENT_FRACTIONS, REPLACEMENT_REPLICATES,
    CIRCUIT_CELL_COUNTS
)


def run_step_1_build_circuit(force=False):
    """Step 1: Build cortical microcircuit configuration."""
    print("\n" + "=" * 70)
    print("STEP 1: Building cortical microcircuit")
    print("=" * 70)

    circuit_dir = RESULTS_DIR / 'circuit'
    circuit_dir.mkdir(parents=True, exist_ok=True)

    total_cells = sum(CIRCUIT_CELL_COUNTS.values())
    print(f"  Cell types: {len(CIRCUIT_CELL_COUNTS)}")
    print(f"  Total cells: {total_cells}")
    for ctype, count in CIRCUIT_CELL_COUNTS.items():
        print(f"    {ctype}: {count}")

    from l5pc.simulation.circuit import build_circuit_config
    config = build_circuit_config()
    print("  DONE: Circuit configuration built.")
    return config


def run_step_2_baseline(force=False):
    """Step 2: Run biophysical baseline (0% replacement)."""
    print("\n" + "=" * 70)
    print("STEP 2: Running biophysical baseline")
    print("=" * 70)

    circuit_dir = RESULTS_DIR / 'circuit'
    baseline_path = circuit_dir / 'baseline_results.json'

    if not force and baseline_path.exists():
        print("  SKIP: Baseline results exist.")
        return

    from l5pc.simulation.circuit import build_circuit_config, run_circuit
    config = build_circuit_config()
    results = run_circuit(config, duration_ms=2000, seed=42)

    from l5pc.utils.io import save_results_json
    save_results_json(results, str(baseline_path))
    print("  DONE: Baseline simulation complete.")


def run_step_3_replacement(n_replicates, force=False):
    """Step 3: Graded surrogate replacement."""
    print("\n" + "=" * 70)
    print("STEP 3: Graded surrogate replacement")
    print(f"  Fractions: {REPLACEMENT_FRACTIONS}")
    print(f"  Replicates per fraction: {n_replicates}")
    print("=" * 70)

    circuit_dir = RESULTS_DIR / 'circuit'
    replacement_path = circuit_dir / 'replacement_results.json'

    if not force and replacement_path.exists():
        print("  SKIP: Replacement results exist.")
        return

    from l5pc.validation.graded_replacement import run_graded_replacement
    results = run_graded_replacement(
        surrogate_dir=str(SURROGATE_DIR),
        output_dir=str(circuit_dir),
        n_replicates=n_replicates
    )
    print(f"  DONE: {len(REPLACEMENT_FRACTIONS)} replacement fractions tested.")


def run_step_4_level1(force=False):
    """Step 4: Level 1 validation - output fidelity."""
    print("\n" + "=" * 70)
    print("STEP 4: Level 1 Validation - Output Fidelity")
    print("=" * 70)

    circuit_dir = RESULTS_DIR / 'circuit'
    l1_path = circuit_dir / 'level1_validation.json'

    if not force and l1_path.exists():
        print("  SKIP: Level 1 results exist.")
        return

    from l5pc.validation.level1_output import run_level1_validation
    results = run_level1_validation(
        bio_dir=str(circuit_dir / 'bio_spikes'),
        surrogate_dir=str(circuit_dir / 'surrogate_spikes')
    )

    from l5pc.utils.io import save_results_json
    save_results_json(results, str(l1_path))
    print(f"  DONE: Level 1 - Cross-condition CC = {results.get('cross_condition_cc', 'N/A')}")


def run_step_5_level2(force=False):
    """Step 5: Level 2 validation - circuit dynamics."""
    print("\n" + "=" * 70)
    print("STEP 5: Level 2 Validation - Circuit Dynamics")
    print("=" * 70)

    circuit_dir = RESULTS_DIR / 'circuit'
    l2_path = circuit_dir / 'level2_validation.json'

    if not force and l2_path.exists():
        print("  SKIP: Level 2 results exist.")
        return

    from l5pc.validation.level2_circuit import run_level2_validation
    results = run_level2_validation(
        bio_results_path=str(circuit_dir / 'baseline_results.json'),
        surrogate_results_path=str(circuit_dir / 'replacement_results.json')
    )

    from l5pc.utils.io import save_results_json
    save_results_json(results, str(l2_path))
    print("  DONE: Level 2 validation complete.")


def run_step_6_level3(force=False):
    """Step 6: Level 3 validation - DESCARTES mechanistic."""
    print("\n" + "=" * 70)
    print("STEP 6: Level 3 Validation - DESCARTES Mechanistic")
    print("=" * 70)

    circuit_dir = RESULTS_DIR / 'circuit'
    l3_path = circuit_dir / 'level3_validation.json'

    if not force and l3_path.exists():
        print("  SKIP: Level 3 results exist.")
        return

    from l5pc.validation.level3_descartes import run_level3_validation
    results = run_level3_validation(
        surrogate_dir=str(SURROGATE_DIR),
        trial_dir=str(circuit_dir),
        results_dir=str(circuit_dir)
    )

    from l5pc.utils.io import save_results_json
    save_results_json(results, str(l3_path))
    print("  DONE: Level 3 - DESCARTES re-probing in circuit context.")


def run_step_7_level4(force=False):
    """Step 7: Level 4 validation - consciousness-relevant metrics."""
    print("\n" + "=" * 70)
    print("STEP 7: Level 4 Validation - Consciousness Metrics")
    print("=" * 70)

    circuit_dir = RESULTS_DIR / 'circuit'
    l4_path = circuit_dir / 'level4_validation.json'

    if not force and l4_path.exists():
        print("  SKIP: Level 4 results exist.")
        return

    from l5pc.validation.level4_consciousness import run_level4_validation
    results = run_level4_validation(
        bio_results_path=str(circuit_dir / 'baseline_results.json'),
        surrogate_results_path=str(circuit_dir / 'replacement_results.json')
    )

    from l5pc.utils.io import save_results_json
    save_results_json(results, str(l4_path))
    print("  DONE: Level 4 validation complete.")
    if 'pci' in results:
        print(f"    PCI bio={results['pci'].get('bio', 'N/A'):.3f}, "
              f"surrogate={results['pci'].get('surrogate', 'N/A'):.3f}")


def run_step_8_visualize(force=False):
    """Step 8: Generate replacement curves and 4-level dashboard."""
    print("\n" + "=" * 70)
    print("STEP 8: Generating Phase 3 visualizations")
    print("=" * 70)

    fig_dir = RESULTS_DIR / 'figures' / 'phase3'
    fig_dir.mkdir(parents=True, exist_ok=True)
    circuit_dir = RESULTS_DIR / 'circuit'

    from l5pc.visualization.replacement_curves import (
        plot_replacement_curve, plot_four_level_dashboard, plot_prediction_test
    )
    from l5pc.utils.io import load_results_json

    # Replacement curves
    replacement_path = circuit_dir / 'replacement_results.json'
    if replacement_path.exists():
        replacement_data = load_results_json(str(replacement_path))
        plot_replacement_curve(
            replacement_data,
            save_path=str(fig_dir / 'replacement_curve.png')
        )

    # 4-level dashboard
    validation_data = {}
    for level in range(1, 5):
        vpath = circuit_dir / f'level{level}_validation.json'
        if vpath.exists():
            validation_data[f'level{level}'] = load_results_json(str(vpath))

    if validation_data:
        plot_four_level_dashboard(
            validation_data,
            save_path=str(fig_dir / 'four_level_dashboard.png')
        )

    # Prediction test
    prediction_path = RESULTS_DIR / 'classification_summary.json'
    if prediction_path.exists():
        classification = load_results_json(str(prediction_path))
        plot_prediction_test(
            classification,
            save_path=str(fig_dir / 'prediction_test.png')
        )

    # Cross-circuit comparison
    from l5pc.analysis.cross_circuit import print_comparison
    print("\n  --- Cross-Circuit Comparison ---")
    print_comparison()

    print(f"  DONE: Phase 3 figures saved to {fig_dir}")


STEPS = {
    1: ('build_circuit', run_step_1_build_circuit),
    2: ('baseline',      run_step_2_baseline),
    3: ('replacement',   run_step_3_replacement),
    4: ('level1',        run_step_4_level1),
    5: ('level2',        run_step_5_level2),
    6: ('level3',        run_step_6_level3),
    7: ('level4',        run_step_7_level4),
    8: ('visualize',     run_step_8_visualize),
}


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Circuit integration')
    parser.add_argument('--start-step', type=int, default=1)
    parser.add_argument('--only-step', type=int, default=None)
    parser.add_argument('--replicates', type=int, default=REPLACEMENT_REPLICATES)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("L5PC DESCARTES - Phase 3: Circuit Integration")
    print(f"  {sum(CIRCUIT_CELL_COUNTS.values())} cells, "
          f"{len(REPLACEMENT_FRACTIONS)} replacement fractions, "
          f"{args.replicates} replicates")
    print("=" * 70)
    t0 = time.time()

    if args.only_step:
        name, func = STEPS[args.only_step]
        if args.only_step == 3:
            func(args.replicates, force=args.force)
        else:
            func(force=args.force)
    else:
        for step_num in range(args.start_step, 9):
            name, func = STEPS[step_num]
            if step_num == 3:
                func(args.replicates, force=args.force)
            else:
                func(force=args.force)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Phase 3 complete in {elapsed / 60:.1f} minutes")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
