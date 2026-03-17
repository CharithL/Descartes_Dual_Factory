#!/usr/bin/env python3
"""
DESCARTES Dual Factory v3.0 -- Circuit 4 Hidden State Extraction

Pre-extracts bin-level and window-level hidden states + biological probe
targets for all 31 human WM patients. Produces self-contained .npy files
that can be uploaded to Vast.ai for the Phase 1 hardened battery without
needing NWB files or the human_wm package on the remote machine.

Outputs per patient (in circuit4_extracted/<patient_id>/):
  hidden_bin_trained.npy      (n_test * T, hidden_size)
  hidden_bin_untrained.npy    (n_test * T, hidden_size)
  hidden_win_trained.npy      (n_test, hidden_size)
  hidden_win_untrained.npy    (n_test, hidden_size)
  probe_targets_bin.npy       (n_test * T, n_targets)  -- targets repeated per bin
  probe_targets_win.npy       (n_test, n_targets)       -- trial-level targets
  probe_target_names.json     list of target names
  patient_meta.json           cc, n_trials, T, input_dim, output_dim, etc.

Run from the L5PC directory:
  cd "L5PC"
  python scripts/extract_circuit4_hidden_states.py

Requires: human_wm package, NWB files in Working memory/data/raw/000469/
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# -- Paths --
PROJECT_ROOT = Path(__file__).parent.parent  # L5PC/
WM_DIR = Path("C:/Users/chari/OneDrive/Documents/Descartes_Cogito/Working memory")
sys.path.insert(0, str(WM_DIR))

CHECKPOINT_DIR = WM_DIR / "data" / "results" / "cross_patient" / "cross_patient"
OUTPUT_DIR = PROJECT_ROOT / "circuit4_extracted"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / 'extraction.log'),
    ],
)
logger = logging.getLogger('c4_extract')

HIDDEN_SIZE = 128
SEED_UNTRAINED = 999


def estimate_task_timing(trial_info, n_bins):
    """Sternberg task phase boundaries: encoding (40%), delay (30%), probe (30%)."""
    encoding_end = int(0.4 * n_bins)
    delay_end = int(0.7 * n_bins)
    return {
        'encoding_bins': slice(0, encoding_end),
        'delay_bins': slice(encoding_end, delay_end),
        'probe_bins': slice(delay_end, n_bins),
    }


def extract_one_patient(patient_dir, nwb_path, schema):
    """Extract hidden states and probe targets for one patient.

    Returns dict with extraction results or None on failure.
    """
    from human_wm.surrogate.models import HumanLSTMSurrogate
    from human_wm.data.nwb_loader import extract_patient_data, split_data
    from human_wm.targets.probe_targets import compute_all_targets

    patient_id = patient_dir.name
    checkpoint_path = patient_dir / f"lstm_h{HIDDEN_SIZE}_s0_best.pt"

    if not checkpoint_path.exists():
        logger.info("  SKIP %s: no checkpoint", patient_id)
        return None

    # Check quality from existing results
    result_path = patient_dir / f"results_lstm_h{HIDDEN_SIZE}_s0.json"
    cc = None
    if result_path.exists():
        with open(result_path) as f:
            prev = json.load(f)
        cc = prev.get('cc', 0)
        if cc < 0.3:
            logger.info("  SKIP %s: CC=%.3f < 0.3", patient_id, cc)
            return None

    logger.info("  %s (CC=%s):", patient_id, f"{cc:.3f}" if cc else "?")

    # -- Load NWB data --
    try:
        X, Y, trial_info = extract_patient_data(nwb_path, schema)
    except Exception as e:
        logger.error("    Error loading NWB: %s", e)
        return None

    splits = split_data(X, Y, trial_info, seed=42)
    X_test = splits['test']['X']       # (n_test, T, n_mtl)
    Y_test = splits['test']['Y']       # (n_test, T, n_frontal)
    trial_info_test = splits['test'].get('trial_info', splits['test'].get('trial_types', {}))

    n_test, T, n_mtl = X_test.shape
    n_frontal = Y_test.shape[2]
    logger.info("    X_test=%s  Y_test=%s  T=%d", X_test.shape, Y_test.shape, T)

    # -- Load trained model --
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    ckpt_input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
    ckpt_output_dim = state_dict['output_proj.weight'].shape[0]

    # Handle dimension mismatches
    if n_mtl != ckpt_input_dim:
        logger.info("    [dim fix] input %d -> %d", n_mtl, ckpt_input_dim)
        if n_mtl > ckpt_input_dim:
            X_test = X_test[:, :, :ckpt_input_dim]
        else:
            pad = np.zeros((n_test, T, ckpt_input_dim - n_mtl))
            X_test = np.concatenate([X_test, pad], axis=2)

    if n_frontal != ckpt_output_dim:
        logger.info("    [dim fix] output %d -> %d", n_frontal, ckpt_output_dim)
        if n_frontal > ckpt_output_dim:
            Y_test = Y_test[:, :, :ckpt_output_dim]
        else:
            pad = np.zeros((n_test, T, ckpt_output_dim - n_frontal))
            Y_test = np.concatenate([Y_test, pad], axis=2)

    input_dim = ckpt_input_dim
    output_dim = ckpt_output_dim

    # -- Forward pass: trained --
    model_trained = HumanLSTMSurrogate(input_dim, output_dim, HIDDEN_SIZE)
    model_trained.load_state_dict(state_dict)
    model_trained.train(False)

    X_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        _, h_seq_trained = model_trained(X_t, return_hidden=True)
        h_trained_3d = h_seq_trained.numpy()  # (n_test, T, hidden_size)

    # -- Forward pass: untrained --
    torch.manual_seed(SEED_UNTRAINED)
    model_untrained = HumanLSTMSurrogate(input_dim, output_dim, HIDDEN_SIZE)
    model_untrained.train(False)

    with torch.no_grad():
        _, h_seq_untrained = model_untrained(X_t, return_hidden=True)
        h_untrained_3d = h_seq_untrained.numpy()  # (n_test, T, hidden_size)

    # -- Bin-level: flatten (n_test * T, hidden_size) --
    h_bin_trained = h_trained_3d.reshape(-1, HIDDEN_SIZE)
    h_bin_untrained = h_untrained_3d.reshape(-1, HIDDEN_SIZE)

    # -- Window-level: trial-averaged (n_test, hidden_size) --
    h_win_trained = h_trained_3d.mean(axis=1)
    h_win_untrained = h_untrained_3d.mean(axis=1)

    logger.info("    Hidden: bin=%s  win=%s", h_bin_trained.shape, h_win_trained.shape)

    # -- Compute biological probe targets --
    task_timing = estimate_task_timing(trial_info_test, T)
    try:
        targets_dict = compute_all_targets(
            Y_test, trial_info_test, task_timing,
            bin_size_s=0.05,
        )
    except Exception as e:
        logger.warning("    compute_all_targets failed: %s -- using fallback targets", e)
        # Fallback: compute what we can directly
        targets_dict = {
            'mean_firing_rate': Y_test.mean(axis=(1, 2)),
            'population_rate': Y_test.sum(axis=2).mean(axis=1),
            'trial_variance': Y_test.var(axis=1).mean(axis=1),
        }

    # Filter out zero-variance or all-NaN targets
    valid_targets = {}
    for name, arr in targets_dict.items():
        arr = np.asarray(arr, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0)
        if arr.std() > 1e-10 and len(arr) == n_test:
            valid_targets[name] = arr

    target_names = sorted(valid_targets.keys())
    n_targets = len(target_names)
    logger.info("    Probe targets: %d valid -- %s", n_targets, target_names)

    if n_targets == 0:
        logger.warning("    SKIP %s: no valid probe targets", patient_id)
        return None

    # -- Build target arrays --
    # Window-level: (n_test, n_targets)
    targets_win = np.column_stack([valid_targets[n] for n in target_names])

    # Bin-level: repeat each trial's target T times -> (n_test * T, n_targets)
    targets_bin = np.repeat(targets_win, T, axis=0)

    # -- Save --
    out_dir = OUTPUT_DIR / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / 'hidden_bin_trained.npy', h_bin_trained.astype(np.float32))
    np.save(out_dir / 'hidden_bin_untrained.npy', h_bin_untrained.astype(np.float32))
    np.save(out_dir / 'hidden_win_trained.npy', h_win_trained.astype(np.float32))
    np.save(out_dir / 'hidden_win_untrained.npy', h_win_untrained.astype(np.float32))
    np.save(out_dir / 'probe_targets_bin.npy', targets_bin.astype(np.float64))
    np.save(out_dir / 'probe_targets_win.npy', targets_win.astype(np.float64))

    with open(out_dir / 'probe_target_names.json', 'w') as f:
        json.dump(target_names, f)

    meta = {
        'patient_id': patient_id,
        'cc': cc,
        'hidden_size': HIDDEN_SIZE,
        'n_test_trials': n_test,
        'T_bins_per_trial': T,
        'n_bin_samples': int(n_test * T),
        'input_dim': input_dim,
        'output_dim': output_dim,
        'n_targets': n_targets,
        'target_names': target_names,
        'bin_size_s': 0.05,
        'sampling_rate_hz': 20.0,  # 1 / 0.05
        'task_timing': {
            'encoding_end_bin': int(0.4 * T),
            'delay_end_bin': int(0.7 * T),
            'total_bins': T,
        },
    }
    with open(out_dir / 'patient_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info("    Saved: %s  (%d bin samples, %d win samples, %d targets)",
                out_dir.name, n_test * T, n_test, n_targets)

    return meta


def find_nwb_for_patient(patient_id, raw_dir):
    """Map patient directory name to NWB file."""
    nwb_name = f"{patient_id}.nwb"
    for nwb in raw_dir.rglob(nwb_name):
        return nwb
    # Fallback: match by sub-ID
    sub_id = patient_id.split('_')[0]
    for nwb in raw_dir.rglob(f"{sub_id}*.nwb"):
        if patient_id in str(nwb):
            return nwb
    return None


def main():
    from human_wm.config import RAW_NWB_DIR, load_nwb_schema

    logger.info("DESCARTES Circuit 4 -- Hidden State Extraction")
    logger.info("Checkpoint dir: %s", CHECKPOINT_DIR)
    logger.info("Output dir: %s", OUTPUT_DIR)

    schema = load_nwb_schema()
    if schema is None:
        logger.error("NWB schema not found")
        sys.exit(1)

    patient_dirs = sorted([d for d in CHECKPOINT_DIR.iterdir() if d.is_dir()])
    logger.info("Found %d patient directories", len(patient_dirs))

    results = {}
    t0 = time.time()

    for i, patient_dir in enumerate(patient_dirs):
        patient_id = patient_dir.name
        logger.info("[%2d/%d] %s", i + 1, len(patient_dirs), patient_id)

        nwb_path = find_nwb_for_patient(patient_id, RAW_NWB_DIR)
        if nwb_path is None:
            logger.info("  SKIP: NWB not found")
            continue

        meta = extract_one_patient(patient_dir, nwb_path, schema)
        if meta is not None:
            results[patient_id] = meta

    elapsed = time.time() - t0
    logger.info("Extraction complete: %d/%d patients in %.1f min",
                len(results), len(patient_dirs), elapsed / 60)

    # Save manifest
    manifest = {
        'n_patients': len(results),
        'patients': results,
        'hidden_size': HIDDEN_SIZE,
        'extraction_time_min': round(elapsed / 60, 1),
    }
    with open(OUTPUT_DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved: %s", OUTPUT_DIR / 'manifest.json')

    # Summary table
    print(f"\n{'='*80}")
    print("Circuit 4 Extraction Summary")
    print(f"{'='*80}")
    print(f"{'Patient':<35} {'CC':>6} {'Trials':>7} {'T':>5} {'Bin N':>7} {'Targets':>8}")
    print("-" * 80)
    for pid, m in sorted(results.items()):
        print(f"{pid:<35} {m['cc'] or 0:>6.3f} {m['n_test_trials']:>7} "
              f"{m['T_bins_per_trial']:>5} {m['n_bin_samples']:>7} {m['n_targets']:>8}")
    print(f"{'='*80}")
    print(f"Total: {len(results)} patients extracted")


if __name__ == '__main__':
    main()
