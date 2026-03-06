"""I/O utilities for trial data and results."""
import json
import numpy as np
from pathlib import Path


def save_trial(trial_dir, trial_idx, inputs, output, level_a, level_b_cond,
               level_b_curr, level_c, metadata=None):
    """Save a single trial's data as compressed numpy archive."""
    path = Path(trial_dir)
    path.mkdir(parents=True, exist_ok=True)
    save_path = path / f'trial_{trial_idx:03d}.npz'
    np.savez_compressed(
        save_path,
        inputs=inputs, output=output,
        level_A_gates=level_a, level_B_cond=level_b_cond,
        level_B_curr=level_b_curr, level_C_emerge=level_c,
    )
    if metadata is not None:
        meta_path = path / 'variable_names.json'
        if not meta_path.exists():
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)


def load_trial(trial_dir, trial_idx):
    """Load a single trial's data."""
    path = Path(trial_dir) / f'trial_{trial_idx:03d}.npz'
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_variable_names(trial_dir):
    """Load variable name metadata."""
    path = Path(trial_dir) / 'variable_names.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_all_trials(trial_dir, split='all'):
    """Load all trials for a given split ('train', 'val', 'test', 'all')."""
    from l5pc.config import TRAIN_SPLIT, VAL_SPLIT, N_TRIALS
    trial_dir = Path(trial_dir)
    if split == 'train':
        indices = range(0, TRAIN_SPLIT)
    elif split == 'val':
        indices = range(TRAIN_SPLIT, TRAIN_SPLIT + VAL_SPLIT)
    elif split == 'test':
        indices = range(TRAIN_SPLIT + VAL_SPLIT, N_TRIALS)
    else:
        indices = range(N_TRIALS)
    trials = []
    for i in indices:
        fpath = trial_dir / f'trial_{i:03d}.npz'
        if fpath.exists():
            trials.append(load_trial(trial_dir, i))
    return trials


def concat_trials(trials, key):
    """Concatenate a specific array across trials along axis 0."""
    arrays = [t[key] for t in trials if key in t]
    if not arrays:
        return np.array([])
    return np.concatenate(arrays, axis=0)


def save_results_json(data, path):
    """Save results dict to JSON with numpy type conversion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        raise TypeError(f"Not JSON serializable: {type(obj)}")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=convert)


def load_results_json(path):
    """Load results from JSON."""
    with open(path) as f:
        return json.load(f)
