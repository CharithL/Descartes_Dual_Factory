"""
L5PC DESCARTES -- Causal Ablation with Progressive Clamping

The DECISIVE test: DeltaR2 and voltage baselines establish that a variable
is decodable and non-trivial, but decoding does not imply causal use.
Progressive clamping answers: does the network actually USE this
representation to compute its output?

Protocol (guide Section 4.8):
  1. Rank hidden dimensions by |correlation| with the target variable.
  2. For increasing k (fraction of total dims):
     a. Clamp the top-k% target-correlated dims to their trial-mean values.
     b. Run a forward pass, measure cross-condition correlation of output.
     c. Clamp k% RANDOM dims (repeat N times), measure output.
     d. Compute z-score = (target_drop - random_mean) / random_std.
  3. If z < CAUSAL_Z_THRESHOLD at any k, the variable is MANDATORY.
  4. The breaking point k classifies redundancy:
     - Concentrated: breaks at < 10% (like g_NMDA_SC at 3.9%)
     - Distributed: breaks at 40-60% (like gamma_amp at h=128)
     - Redundant: breaks at > 70% (like gamma_amp at h=512)
"""

import logging
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from l5pc.config import (
    ABLATION_K_FRACTIONS,
    ABLATION_N_RANDOM,
    CAUSAL_Z_THRESHOLD,
    DELTA_THRESHOLD_LEARNED,
    HIDDEN_SIZES,
)
from l5pc.utils.io import load_results_json, save_results_json
from l5pc.utils.metrics import cross_condition_correlation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forward pass with hidden-dimension clamping
# ---------------------------------------------------------------------------

def forward_with_clamp(model, test_inputs, clamp_dims, hidden_means):
    """Run LSTM forward pass with specified hidden dims clamped to their mean.

    At each timestep, after the LSTM computes the hidden state, the
    specified dimensions are overwritten with their pre-computed means.
    This severs the information carried by those dimensions.

    Parameters
    ----------
    model : L5PC_LSTM
        Trained surrogate model (set to evaluation mode before calling).
    test_inputs : torch.Tensor
        Shape (n_trials, T, input_dim).
    clamp_dims : list or ndarray of int
        Indices of hidden dimensions to clamp.
    hidden_means : ndarray, shape (hidden_size,)
        Mean activation per hidden dimension (computed over training set).

    Returns
    -------
    output : ndarray, shape (n_trials, T)
        Model output (spike probability) with clamped hidden states.
    """
    device = next(model.parameters()).device
    test_inputs = test_inputs.to(device)
    clamp_dims = list(clamp_dims)

    if len(clamp_dims) == 0:
        # No clamping -- standard forward pass
        with torch.no_grad():
            result = model(test_inputs)
            out = result[0] if isinstance(result, tuple) else result
            return out.cpu().numpy()

    # Manual step-by-step forward to allow per-timestep clamping
    batch_size, T, _ = test_inputs.shape
    hidden_size = model.hidden_size
    n_layers = model.n_layers

    # Initialise LSTM hidden and cell states
    h = torch.zeros(n_layers, batch_size, hidden_size, device=device)
    c = torch.zeros(n_layers, batch_size, hidden_size, device=device)

    # Mean values as tensor for clamping
    mean_tensor = torch.tensor(
        hidden_means, dtype=torch.float32, device=device
    )

    outputs = []
    with torch.no_grad():
        for t in range(T):
            x_t = test_inputs[:, t:t+1, :]  # (batch, 1, input_dim)
            lstm_out, (h, c) = model.lstm(x_t, (h, c))
            # lstm_out: (batch, 1, hidden_size)

            # Clamp specified dims in the LAST layer's hidden state
            h_last = h[-1]  # (batch, hidden_size)
            h_last[:, clamp_dims] = mean_tensor[clamp_dims].unsqueeze(0)
            h[-1] = h_last

            # Readout from (clamped) last-layer hidden state
            logit = model.readout(h_last)  # (batch, 1)
            prob = torch.sigmoid(logit).squeeze(-1)  # (batch,)
            outputs.append(prob)

    output = torch.stack(outputs, dim=1)  # (batch, T)
    return output.cpu().numpy()


# ---------------------------------------------------------------------------
# Progressive ablation
# ---------------------------------------------------------------------------

def causal_ablation(model, test_inputs, test_outputs, target_y,
                    hidden_states, k_fractions=None, n_random_repeats=None):
    """Progressive ablation protocol.

    Parameters
    ----------
    model : L5PC_LSTM
        Trained surrogate (must be set to evaluation mode before calling).
    test_inputs : torch.Tensor, shape (n_trials, T, input_dim)
    test_outputs : ndarray, shape (n_trials, T)
        Ground-truth output for computing cross-condition correlation.
    target_y : ndarray, shape (n_trials,)
        Trial-level biophysical target variable.
    hidden_states : ndarray, shape (n_trials, hidden_size)
        Trial-averaged hidden states.
    k_fractions : list of float, optional
    n_random_repeats : int, optional

    Returns
    -------
    results : list of dict
        One entry per k value with keys:
        k_frac, n_clamped, target_cc, random_cc_mean, random_cc_std,
        z_score, verdict.
    baseline_cc : float
        Cross-condition correlation with NO clamping (intact model).
    """
    if k_fractions is None:
        k_fractions = ABLATION_K_FRACTIONS
    if n_random_repeats is None:
        n_random_repeats = ABLATION_N_RANDOM

    hidden_size = hidden_states.shape[1]
    hidden_means = np.mean(hidden_states, axis=0)

    # Rank dims by |correlation| with target variable
    correlations = np.array([
        abs(float(stats.pearsonr(hidden_states[:, d], target_y)[0]))
        if np.std(hidden_states[:, d]) > 1e-10 else 0.0
        for d in range(hidden_size)
    ])
    sorted_dims = np.argsort(correlations)[::-1]  # Descending |r|

    # Baseline: intact model output cross-condition correlation
    intact_output = forward_with_clamp(model, test_inputs, [], hidden_means)
    baseline_cc = cross_condition_correlation(intact_output, test_outputs)

    rng = np.random.RandomState(42)
    results = []

    for k_frac in k_fractions:
        n_clamp = max(1, int(round(k_frac * hidden_size)))

        # --- Target-correlated clamping ---
        target_dims = sorted_dims[:n_clamp]
        target_output = forward_with_clamp(model, test_inputs, target_dims,
                                           hidden_means)
        target_cc = cross_condition_correlation(target_output, test_outputs)

        # --- Random clamping (repeated) ---
        random_ccs = []
        for _ in range(n_random_repeats):
            rand_dims = rng.choice(hidden_size, size=n_clamp, replace=False)
            rand_output = forward_with_clamp(model, test_inputs, rand_dims,
                                             hidden_means)
            rand_cc = cross_condition_correlation(rand_output, test_outputs)
            random_ccs.append(rand_cc)

        random_mean = float(np.mean(random_ccs))
        random_std = float(np.std(random_ccs))

        # z-score: how much worse is target-clamping vs random?
        if random_std > 1e-10:
            z_score = (target_cc - random_mean) / random_std
        else:
            # All random ablations identical -- use sign heuristic
            z_score = -10.0 if target_cc < random_mean else 0.0

        # Verdict at this k
        if z_score < CAUSAL_Z_THRESHOLD:
            verdict = 'CAUSAL'
        else:
            verdict = 'NON_CAUSAL'

        entry = {
            'k_frac': float(k_frac),
            'n_clamped': int(n_clamp),
            'target_cc': float(target_cc),
            'target_cc_drop': float(baseline_cc - target_cc),
            'random_cc_mean': random_mean,
            'random_cc_std': random_std,
            'random_ccs': [float(x) for x in random_ccs],
            'z_score': float(z_score),
            'verdict': verdict,
        }
        results.append(entry)
        logger.info(
            "  k=%.0f%% (%d dims): target_cc=%.3f  random_cc=%.3f +/- %.3f  "
            "z=%.2f  [%s]",
            k_frac * 100, n_clamp, target_cc, random_mean, random_std,
            z_score, verdict,
        )

    return results, float(baseline_cc)


# ---------------------------------------------------------------------------
# Redundancy type classification
# ---------------------------------------------------------------------------

def classify_mandatory_type(ablation_results, baseline_cc):
    """Classify redundancy type based on where the model breaks.

    Parameters
    ----------
    ablation_results : list of dict
        Output from causal_ablation().
    baseline_cc : float
        Intact model cross-condition correlation.

    Returns
    -------
    classification : str
        One of: 'NON_CAUSAL', 'MANDATORY_CONCENTRATED',
        'MANDATORY_DISTRIBUTED', 'MANDATORY_REDUNDANT'.
    breaking_point : float or None
        k_frac at which z < CAUSAL_Z_THRESHOLD (first occurrence).
    """
    causal_entries = [r for r in ablation_results if r['verdict'] == 'CAUSAL']

    if not causal_entries:
        return 'NON_CAUSAL', None

    # Breaking point: smallest k where causality is detected
    breaking_point = min(r['k_frac'] for r in causal_entries)

    if breaking_point <= 0.10:
        return 'MANDATORY_CONCENTRATED', breaking_point
    elif breaking_point <= 0.60:
        return 'MANDATORY_DISTRIBUTED', breaking_point
    else:
        return 'MANDATORY_REDUNDANT', breaking_point


# ---------------------------------------------------------------------------
# Full ablation runner
# ---------------------------------------------------------------------------

def _set_model_eval(model):
    """Set model to evaluation mode (disable dropout, batch norm training)."""
    model.train(False)
    return model


def run_all_ablations(ridge_results_dir, hidden_dir, model_dir,
                      test_data_dir, delta_threshold=None, save_path=None):
    """Run ablation on every variable with DeltaR2 > threshold.

    Parameters
    ----------
    ridge_results_dir : str or Path
        Directory with ridge result JSONs.
    hidden_dir : str or Path
        Directory with hidden-state .npy files.
    model_dir : str or Path
        Directory with saved model checkpoints.
    test_data_dir : str or Path
        Directory with test set trial data.
    delta_threshold : float, optional
        Minimum DeltaR2 to trigger ablation. Default: DELTA_THRESHOLD_LEARNED.
    save_path : str or Path, optional
        Output JSON path.

    Returns
    -------
    all_ablation_results : dict
    """
    if delta_threshold is None:
        delta_threshold = DELTA_THRESHOLD_LEARNED

    ridge_results_dir = Path(ridge_results_dir)
    hidden_dir = Path(hidden_dir)
    model_dir = Path(model_dir)
    test_data_dir = Path(test_data_dir)

    # Load test data
    from l5pc.utils.io import load_all_trials
    test_trials = load_all_trials(test_data_dir, split='test')
    if not test_trials:
        raise FileNotFoundError(
            f"No test trials found in {test_data_dir}"
        )

    # Stack test inputs and outputs
    test_inputs_np = np.stack([t['inputs'] for t in test_trials])
    test_outputs_np = np.stack([t['output'] for t in test_trials])

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Ablation device: %s", device)

    all_results = {}

    for hs in HIDDEN_SIZES:
        # Load model -- try both naming conventions from training
        model_path = model_dir / f'lstm_h{hs}_best.pt'
        if not model_path.exists():
            model_path = model_dir / f'lstm_h{hs}.pt'
        if not model_path.exists():
            logger.warning("Model not found for h=%d, skipping", hs)
            continue

        from l5pc.surrogates.lstm import L5PC_LSTM
        model = L5PC_LSTM(hidden_size=hs)
        state_dict = torch.load(model_path, map_location=device,
                                weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        _set_model_eval(model)

        # Load hidden states via shared loader (handles .npz and trial-averaging)
        from l5pc.probing.ridge_probe import _load_hidden_states
        try:
            hidden_states = _load_hidden_states(hidden_dir, hs, trained=True)
        except FileNotFoundError as e:
            logger.warning("Hidden states not found for h=%d: %s", hs, e)
            continue

        # Convert test inputs to tensor
        test_inputs = torch.tensor(test_inputs_np, dtype=torch.float32)

        # Scan ridge results for qualifying variables
        for level in ['A', 'B', 'C']:
            ridge_path = ridge_results_dir / f'ridge_level{level}_h{hs}.json'
            if not ridge_path.exists():
                continue

            ridge_data = load_results_json(ridge_path)

            for r in ridge_data.get('results', []):
                if r.get('delta_R2', 0) < delta_threshold:
                    continue

                var_name = r['var_name']
                logger.info("Ablation: %s  level=%s  h=%d",
                            var_name, level, hs)

                # Load target variable for test trials
                target_y = _load_target_for_test(
                    test_data_dir, level, var_name, test_trials
                )
                if target_y is None:
                    logger.warning("  Target %s not found, skipping",
                                   var_name)
                    continue

                # Ensure trial count alignment
                n = min(len(target_y), hidden_states.shape[0],
                        test_inputs.shape[0])

                ablation_results, baseline_cc = causal_ablation(
                    model, test_inputs[:n], test_outputs_np[:n],
                    target_y[:n], hidden_states[:n],
                )

                classification, breaking_point = classify_mandatory_type(
                    ablation_results, baseline_cc
                )

                key = f'{level}_{var_name}_h{hs}'
                all_results[key] = {
                    'var_name': var_name,
                    'level': level,
                    'hidden_size': hs,
                    'baseline_cc': baseline_cc,
                    'classification': classification,
                    'breaking_point': breaking_point,
                    'ablation_steps': ablation_results,
                }

                logger.info("  -> %s (breaking at %.0f%%)",
                            classification,
                            (breaking_point or 0) * 100)

    # Save
    if save_path is None:
        save_path = ridge_results_dir / 'ablation_results.json'

    output = {
        'n_ablated': len(all_results),
        'n_causal': sum(
            1 for r in all_results.values()
            if r['classification'] != 'NON_CAUSAL'
        ),
        'n_non_causal': sum(
            1 for r in all_results.values()
            if r['classification'] == 'NON_CAUSAL'
        ),
        'results': all_results,
    }
    save_results_json(output, save_path)
    logger.info("Saved ablation results to %s", save_path)
    return output


def _load_target_for_test(test_data_dir, level, var_name, test_trials):
    """Load a specific target variable from test trials.

    Returns ndarray (n_test_trials,) or None if not found.
    """
    level_key_map = {
        'A': 'level_A_gates',
        'B': 'level_B_cond',
        'C': 'level_C_emerge',
    }
    data_key = level_key_map.get(level)
    if data_key is None:
        return None

    # Try loading from trial data
    values = []
    for t in test_trials:
        if data_key not in t:
            return None
        arr = t[data_key]
        if arr.ndim == 1:
            # Single variable
            values.append(float(np.mean(arr)))
        elif arr.ndim == 2:
            # Multiple variables -- need to find column index
            from l5pc.utils.io import load_variable_names
            var_meta = load_variable_names(test_data_dir)
            if var_meta and f'level_{level}' in var_meta:
                names = var_meta[f'level_{level}']
            elif var_meta:
                # Try alternate key patterns
                names = None
                for mk in var_meta:
                    if level.lower() in mk.lower():
                        names = var_meta[mk]
                        break
                if names is None:
                    names = [f'var_{i}' for i in range(arr.shape[1])]
            else:
                names = [f'var_{i}' for i in range(arr.shape[1])]

            if var_name in names:
                col_idx = names.index(var_name)
                values.append(float(np.mean(arr[:, col_idx])))
            else:
                return None

    if not values:
        return None
    return np.array(values)
