"""
L5PC DESCARTES -- Orchestrator for Bahl Model Simulation Trials

Runs all 500 Bahl-model trials with full multi-level recording:

    1. Generate synaptic inputs for all conditions (stimulation.py).
    2. For each trial:
       a. Instantiate a fresh BahlCell.
       b. Attach synapses driven by the trial's input spike trains.
       c. Set up NEURON recording vectors (recording.py).
       d. Run NEURON simulation for SIM_DURATION_MS.
       e. Extract and downsample recordings to 0.5 ms resolution.
       f. Compute Level B (effective conductances) and Level C (emergent
          properties) post-hoc from the recorded traces.
       g. Save the trial to disk via utils.io.save_trial.
    3. Skip trials whose output files already exist (for restartability).

Usage
-----
    from l5pc.simulation.run_bahl_sim import run_all_trials
    run_all_trials(n_trials=500, output_dir='data/bahl_trials')

Or from the command line:
    python -m l5pc.simulation.run_bahl_sim
"""
import time
import numpy as np
from pathlib import Path

from l5pc.config import (
    N_TRIALS,
    SIM_DURATION_MS,
    NEURON_DT_MS,
    RECORDING_DT_MS,
    DOWNSAMPLE_FACTOR,
    T_STEPS,
    BAHL_TRIAL_DIR,
    BAHL_REGIONS,
    STIM_CONDITIONS,
    N_BASAL_SYN,
    N_APICAL_SYN,
    N_SOMA_SYN,
)

from l5pc.simulation.stimulation import generate_all_trials
from l5pc.simulation.bahl_model import BahlCell
from l5pc.simulation.recording import (
    setup_recordings,
    extract_recordings,
    get_variable_names,
    flatten_level_a,
    flatten_level_b_currents,
    flatten_voltages,
)
from l5pc.utils.io import save_trial


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_trials(n_trials=None, output_dir=None, seed=42):
    """Run all Bahl simulation trials and save results to disk.

    Parameters
    ----------
    n_trials : int, optional
        Number of trials to run. Defaults to config.N_TRIALS (500).
    output_dir : str or Path, optional
        Directory to write trial .npz files. Defaults to config.BAHL_TRIAL_DIR.
    seed : int, optional
        Random seed for reproducible input generation. Default 42.
    """
    if n_trials is None:
        n_trials = N_TRIALS
    if output_dir is None:
        output_dir = BAHL_TRIAL_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Late imports for Level B / Level C post-hoc computation
    from l5pc.analysis.effective_conductances import compute_effective_conductances
    from l5pc.analysis.emergent_properties import compute_emergent_properties

    # ------------------------------------------------------------------
    # Step 1: Generate all trial inputs
    # ------------------------------------------------------------------
    print(f"[run_bahl_sim] Generating {n_trials} trial inputs (seed={seed})...")
    all_inputs = generate_all_trials(
        stim_conditions=STIM_CONDITIONS,
        duration_ms=SIM_DURATION_MS,
        seed=seed,
    )

    # Truncate if n_trials < total generated
    all_inputs = all_inputs[:n_trials]
    print(f"[run_bahl_sim] Generated {len(all_inputs)} trial inputs across "
          f"{len(STIM_CONDITIONS)} conditions.")

    # ------------------------------------------------------------------
    # Step 2: Initialise NEURON
    # ------------------------------------------------------------------
    from neuron import h
    h.load_file('stdrun.hoc')

    # Variable-name metadata (computed once from first cell)
    var_names = None
    t_start_all = time.time()

    # ------------------------------------------------------------------
    # Step 3: Run each trial
    # ------------------------------------------------------------------
    for trial_idx, input_dict in enumerate(all_inputs):

        # --- Check if output already exists (restartability) ---
        trial_path = output_dir / f'trial_{trial_idx:03d}.npz'
        if trial_path.exists():
            print(f"  [trial {trial_idx:3d}/{n_trials}] "
                  f"Already exists, skipping.")
            continue

        t0 = time.time()
        condition = input_dict['condition']

        # --- 2a. Create fresh cell ---
        cell = BahlCell()

        # --- 2b. Attach synapses ---
        cell.create_synapses(input_dict)

        # --- 2c. Set up recordings ---
        rec = setup_recordings(cell)

        # Capture variable names on first run
        if var_names is None:
            var_names = get_variable_names(rec)

        # --- 2d. Run NEURON ---
        h.dt = NEURON_DT_MS
        h.celsius = 37.0
        h.v_init = -75.0
        h.tstop = SIM_DURATION_MS
        h.finitialize(h.v_init)
        h.continuerun(h.tstop)

        # --- 2e. Extract and downsample ---
        extracted = extract_recordings(rec, downsample_factor=DOWNSAMPLE_FACTOR)

        # --- Build input array for storage ---
        # Downsample binary spike trains to match recording resolution
        basal_ds = input_dict['basal'][::DOWNSAMPLE_FACTOR, :]
        apical_ds = input_dict['apical'][::DOWNSAMPLE_FACTOR, :]
        soma_ds = input_dict['soma'][::DOWNSAMPLE_FACTOR, :]

        # Ensure consistent length (T_STEPS)
        basal_ds = _pad_or_trim(basal_ds, T_STEPS)
        apical_ds = _pad_or_trim(apical_ds, T_STEPS)
        soma_ds = _pad_or_trim(soma_ds, T_STEPS)

        # Concatenate into single input array (T, TOTAL_SYN)
        inputs = np.concatenate(
            [basal_ds, apical_ds, soma_ds], axis=1
        ).astype(np.float32)

        # --- Output: somatic voltage ---
        output = extracted.get('soma', {}).get('v', np.zeros(T_STEPS))
        output = _pad_or_trim_1d(output, T_STEPS).astype(np.float32)

        # --- 2f. Level A: gating variables ---
        level_a = flatten_level_a(extracted, var_names)
        level_a = _pad_or_trim(level_a, T_STEPS)

        # --- Level B: effective conductances (post-hoc from gates + gbar) ---
        gbar_dict = cell.get_gbar_values()

        # Build flat gates dict: 'm_NaTa_t_soma' -> array
        flat_gates = {}
        for _region in BAHL_REGIONS:
            if _region not in extracted:
                continue
            for _chan, _gate_dict in extracted[_region].get('gates', {}).items():
                for _gate_name, _gate_arr in _gate_dict.items():
                    flat_gates[f'{_gate_name}_{_chan}_{_region}'] = _gate_arr

        level_b_cond_dict = compute_effective_conductances(
            flat_gates, gbar_dict
        )
        # Convert dict -> 2D array (T, N_conductances)
        if level_b_cond_dict:
            cond_keys = sorted(level_b_cond_dict.keys())
            level_b_cond = np.column_stack(
                [level_b_cond_dict[k] for k in cond_keys]
            )
        else:
            level_b_cond = np.empty((T_STEPS, 0), dtype=np.float32)
        level_b_cond = _pad_or_trim(level_b_cond, T_STEPS)

        # --- Level B: ionic currents ---
        level_b_curr = flatten_level_b_currents(extracted, var_names)
        level_b_curr = _pad_or_trim(level_b_curr, T_STEPS)

        # --- Level C: emergent properties (post-hoc) ---
        # Build flat recordings dict for compute_emergent_properties
        emerge_rec = {}
        if 'soma' in extracted and 'v' in extracted['soma']:
            emerge_rec['V_soma'] = extracted['soma']['v']
        if 'tuft' in extracted and 'v' in extracted['tuft']:
            emerge_rec['V_tuft'] = extracted['tuft']['v']
        if 'nexus' in extracted and 'v' in extracted['nexus']:
            emerge_rec['V_nexus'] = extracted['nexus']['v']
        if 'nexus' in extracted:
            cai = extracted['nexus'].get('cai')
            if cai is not None:
                emerge_rec['cai_nexus'] = cai
        if 'tuft' in extracted:
            cai = extracted['tuft'].get('cai')
            if cai is not None:
                emerge_rec['cai_tuft'] = cai
        if 'soma' in extracted:
            for _cn, _ca in extracted['soma'].get('currents', {}).items():
                if _cn == 'ina':
                    emerge_rec['I_NaTa_t_soma'] = _ca

        level_c_dict = compute_emergent_properties(
            emerge_rec, dt_ms=RECORDING_DT_MS
        )
        # Convert dict of scalars -> 1D array for npz storage
        level_c_keys = sorted(level_c_dict.keys())
        level_c = np.array(
            [float(level_c_dict[k]) for k in level_c_keys],
            dtype=np.float32
        )

        # --- 2g. Save trial ---
        metadata = {
            **var_names,
            'level_B_cond_keys': cond_keys if level_b_cond_dict else [],
            'level_C_keys': level_c_keys,
            'condition_labels': {
                trial_idx: condition
            },
            'rates': {
                'basal_hz': input_dict.get('basal_rate', 0.0),
                'apical_hz': input_dict.get('apical_rate', 0.0),
                'soma_hz': input_dict.get('soma_rate', 0.0),
            },
        }

        save_trial(
            trial_dir=output_dir,
            trial_idx=trial_idx,
            inputs=inputs,
            output=output,
            level_a=level_a,
            level_b_cond=level_b_cond,
            level_b_curr=level_b_curr,
            level_c=level_c,
            metadata=metadata,
        )

        elapsed = time.time() - t0
        print(f"  [trial {trial_idx:3d}/{n_trials}] "
              f"condition={condition:<14s}  "
              f"basal={input_dict.get('basal_rate', 0):.1f} Hz  "
              f"apical={input_dict.get('apical_rate', 0):.1f} Hz  "
              f"spikes={int(np.sum(output > 0)):3d}  "
              f"time={elapsed:.1f}s")

        # Clean up NEURON objects to free memory between trials
        _cleanup_neuron(cell, h)

    total_time = time.time() - t_start_all
    print(f"\n[run_bahl_sim] Completed {n_trials} trials in "
          f"{total_time:.1f}s ({total_time/max(n_trials,1):.2f}s/trial).")
    print(f"[run_bahl_sim] Output saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_or_trim(arr, target_rows):
    """Ensure a 2-D array has exactly target_rows rows."""
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]
    if n == target_rows:
        return arr.astype(np.float32)
    if n > target_rows:
        return arr[:target_rows].astype(np.float32)
    # Pad with zeros
    pad = np.zeros((target_rows - n, arr.shape[1]), dtype=np.float32)
    return np.vstack([arr, pad]).astype(np.float32)


def _pad_or_trim_1d(arr, target_len):
    """Ensure a 1-D array has exactly target_len elements."""
    arr = np.asarray(arr).ravel()
    n = len(arr)
    if n == target_len:
        return arr.astype(np.float32)
    if n > target_len:
        return arr[:target_len].astype(np.float32)
    return np.concatenate([arr, np.zeros(target_len - n)]).astype(np.float32)


def _cleanup_neuron(cell, h):
    """Delete NEURON objects to prevent memory leaks between trials."""
    # Clear synapse / netcon references
    cell.synapses.clear()
    cell._netcons.clear()
    cell._netstims.clear()

    # Delete sections
    for region, sec in cell.sections.items():
        try:
            h.delete_section(sec=sec)
        except Exception:
            pass
    cell.sections.clear()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run Bahl L5PC simulation trials for DESCARTES.'
    )
    parser.add_argument(
        '--n-trials', type=int, default=N_TRIALS,
        help=f'Number of trials to run (default: {N_TRIALS})'
    )
    parser.add_argument(
        '--output-dir', type=str, default=str(BAHL_TRIAL_DIR),
        help=f'Output directory (default: {BAHL_TRIAL_DIR})'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for input generation (default: 42)'
    )
    args = parser.parse_args()

    run_all_trials(
        n_trials=args.n_trials,
        output_dir=args.output_dir,
        seed=args.seed,
    )
