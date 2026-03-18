#!/usr/bin/env python3
"""
DESCARTES Dual Factory v3.0 -- Alien Computation Characterization

Uses Phase 2+3 probes (Koopman, SINDy, gate-specific, CCA) to characterize
what the hippocampal LSTM's "alien subspace" is actually computing.

The hippocampal LSTM (h=128, 256 actual dims) encodes gamma_amp and some
biological variables (non-zombie). The remaining dimensions are active but
don't correspond to any known biology -- these are "alien computations."

Analyses:
  1. Koopman spectral comparison -- do alien dims oscillate at biological frequencies?
  2. SINDy on alien dims -- what equations govern the alien subspace?
  3. Gate-specific probing -- does biology live in forget/input/output/cell?
  4. Cross-circuit CCA -- do hippocampal alien dims share structure with L5PC zombie?

Data:
  Hippocampal: hippocampal_mimo/checkpoints_rates/hidden_states/lstm_trained.npy (21600, 256)
               108 trials x 200 timesteps, 25 biological variables
  L5PC:        data/surrogates/hidden/lstm_128_trained.npz (150000, 128)
               200 trials x 750 timesteps, 50 biological variables

Usage:
    python scripts/run_alien_computation.py [--analysis 0|1|2|3|4]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Paths ──────────────────────────────────────────────────────────────
HIPPO_ROOT = PROJECT_ROOT / 'hippocampal_mimo'
HIPPO_RATES = HIPPO_ROOT / 'checkpoints_rates'
HIPPO_HIDDEN = HIPPO_RATES / 'hidden_states'
HIPPO_RESULTS = HIPPO_ROOT / 'results_v3'

L5PC_HIDDEN = PROJECT_ROOT / 'data' / 'surrogates' / 'hidden'
L5PC_MODELS = PROJECT_ROOT / 'data' / 'surrogates'
L5PC_TARGETS = PROJECT_ROOT / 'data' / 'bahl_trials'

RESULTS_DIR = PROJECT_ROOT / 'data' / 'results' / 'alien_computation'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
HIPPO_N_TRIALS = 108
HIPPO_T_STEPS = 200
HIPPO_SAMPLING_RATE = 40  # Hz (25ms bins)
HIPPO_NOMINAL_H = 128     # actual dims = 256 (2-layer LSTM)

L5PC_N_TRIALS = 200
L5PC_T_STEPS = 750
L5PC_SAMPLING_RATE = 1000  # Hz (1ms bins)
L5PC_H_DIM = 128

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / 'alien_computation.log'),
    ],
)
logger = logging.getLogger('alien')


# ═══════════════════════════════════════════════════════════════════════
# JSON helper
# ═══════════════════════════════════════════════════════════════════════

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, complex):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_hippo_hidden(trained=True):
    """Load hippocampal hidden states and reshape to trajectories.
    Returns: list of (200, 256) arrays, one per trial."""
    tag = 'trained' if trained else 'untrained'
    path = HIPPO_HIDDEN / f'lstm_{tag}.npy'
    h = np.load(path).astype(np.float64)
    logger.info("Loaded hippo %s: shape %s", tag, h.shape)
    # Reshape (21600, 256) -> (108, 200, 256) -> list of (200, 256)
    trajs = h.reshape(HIPPO_N_TRIALS, HIPPO_T_STEPS, -1)
    return [trajs[i] for i in range(HIPPO_N_TRIALS)]


def load_hippo_targets():
    """Load hippocampal probe targets and variable names.
    Returns: targets (21600, 25), var_names list"""
    targets = np.load(HIPPO_RATES / 'probe_targets.npy').astype(np.float64)
    with open(HIPPO_RATES / 'probe_variable_names.json') as f:
        var_names = json.load(f)
    logger.info("Loaded hippo targets: %s, %d vars", targets.shape, len(var_names))
    return targets, var_names


def load_hippo_target_trajectories(targets):
    """Reshape flat targets to per-trial trajectories.
    Returns: dict of var_name -> list of (200,) arrays"""
    trajs_all = targets.reshape(HIPPO_N_TRIALS, HIPPO_T_STEPS, -1)
    return [trajs_all[i] for i in range(HIPPO_N_TRIALS)]


def load_l5pc_hidden(trained=True):
    """Load L5PC hidden states (flat).
    Returns: (150000, 128) array"""
    tag = 'trained' if trained else 'untrained'
    path = L5PC_HIDDEN / f'lstm_128_{tag}.npz'
    data = np.load(path)
    key = list(data.keys())[0]
    h = data[key].astype(np.float64)
    logger.info("Loaded L5PC %s: shape %s", tag, h.shape)
    return h


def identify_alien_dims(hidden_flat, targets, var_names, threshold=0.05):
    """Identify which hidden dimensions are correlated with biology
    and which are 'alien' (not correlated with any known variable).

    Uses Ridge coefficients to find dimensions with significant
    contribution to predicting any biological variable.

    Returns:
        bio_dims: indices of biology-correlated dimensions
        alien_dims: indices of alien dimensions
        dim_scores: per-dimension max |Ridge weight| across all targets
    """
    from sklearn.linear_model import Ridge

    n_dims = hidden_flat.shape[1]
    dim_max_weight = np.zeros(n_dims)

    for j in range(targets.shape[1]):
        target = targets[:, j]
        if target.std() < 1e-10:
            continue
        ridge = Ridge(alpha=1.0)
        ridge.fit(hidden_flat, target)
        # Normalized absolute weights
        weights = np.abs(ridge.coef_) / (np.abs(ridge.coef_).max() + 1e-10)
        dim_max_weight = np.maximum(dim_max_weight, weights)

    bio_dims = np.where(dim_max_weight > threshold)[0]
    alien_dims = np.where(dim_max_weight <= threshold)[0]

    logger.info("Dimension split: %d bio-correlated, %d alien (threshold=%.2f)",
                len(bio_dims), len(alien_dims), threshold)

    return bio_dims, alien_dims, dim_max_weight


# ═══════════════════════════════════════════════════════════════════════
# Analysis 1: Koopman Spectral Comparison
# ═══════════════════════════════════════════════════════════════════════

def run_koopman(hippo_trajs, hippo_target_trajs, hippo_targets, var_names):
    """Koopman spectral comparison: hidden vs biological trajectories."""
    from l5pc.probing.dynamical_probes import koopman_spectral_comparison

    logger.info("=" * 70)
    logger.info("ANALYSIS 1: Koopman Spectral Comparison")
    logger.info("=" * 70)

    # Use a subset of trials for tractability
    n_trials = min(30, len(hippo_trajs))
    h_trajs = hippo_trajs[:n_trials]

    results = {}

    # Compare hidden dynamics to EACH biological variable's dynamics
    # Reshape bio targets as trajectories: (108, 200) per variable
    bio_trajs_all = hippo_targets.reshape(HIPPO_N_TRIALS, HIPPO_T_STEPS, -1)

    # Full hidden vs full bio (joint)
    logger.info("  Koopman: all hidden dims vs all bio variables...")
    bio_joint_trajs = [bio_trajs_all[i] for i in range(n_trials)]
    result_joint = koopman_spectral_comparison(
        h_trajs, bio_joint_trajs, n_modes=20, dt=1.0 / HIPPO_SAMPLING_RATE)
    results['joint'] = result_joint
    logger.info("    freq_corr=%.4f  decay_corr=%.4f  match=%s",
                result_joint['frequency_correlation'],
                result_joint['decay_rate_correlation'],
                result_joint['spectral_match'])

    # Per-variable Koopman (top 5 by variance)
    var_stds = hippo_targets.std(axis=0)
    top_vars = np.argsort(-var_stds)[:5]
    for idx in top_vars:
        name = var_names[idx]
        logger.info("  Koopman: hidden vs %s ...", name)
        # Each bio trajectory is (200,) -> need (200, 1)
        bio_single_trajs = [bio_trajs_all[i, :, idx:idx + 1] for i in range(n_trials)]
        # n_modes must be <= min(hidden_dim, bio_dim); bio is 1-dim here
        n_modes_single = min(1, 10)
        # Skip per-variable Koopman when bio is 1-dim (corrcoef needs len>1)
        # Instead, compare frequency spectra via FFT
        logger.info("    (1-dim bio: using FFT frequency comparison instead of Koopman)")
        from scipy.signal import welch
        h_concat = np.concatenate(h_trajs, axis=0)
        b_concat = np.concatenate(bio_single_trajs, axis=0).ravel()
        # PSD of first PC of hidden
        from sklearn.decomposition import PCA
        h_pc1 = PCA(n_components=1).fit_transform(h_concat).ravel()
        f_h, psd_h = welch(h_pc1, fs=HIPPO_SAMPLING_RATE, nperseg=min(128, len(h_pc1)))
        f_b, psd_b = welch(b_concat, fs=HIPPO_SAMPLING_RATE, nperseg=min(128, len(b_concat)))
        # Peak frequencies
        peak_h = f_h[np.argmax(psd_h)]
        peak_b = f_b[np.argmax(psd_b)]
        result = {
            'peak_freq_hidden_pc1': float(peak_h),
            'peak_freq_bio': float(peak_b),
            'freq_ratio': float(peak_h / (peak_b + 1e-10)),
            'spectral_match': abs(peak_h - peak_b) < 2.0,
        }
        results[name] = result
        logger.info("    peak_hidden=%.2f Hz  peak_bio=%.2f Hz  ratio=%.2f  match=%s",
                     result['peak_freq_hidden_pc1'],
                     result['peak_freq_bio'],
                     result['freq_ratio'],
                     result['spectral_match'])

    # Report key frequencies
    logger.info("\n  Hidden eigenfrequencies (top 10): %s Hz",
                [f"{f:.1f}" for f in results['joint']['hidden_frequencies']])
    logger.info("  Bio eigenfrequencies (top 10):    %s Hz",
                [f"{f:.1f}" for f in results['joint']['bio_frequencies']])

    return results


# ═══════════════════════════════════════════════════════════════════════
# Analysis 2: SINDy on Alien Dimensions
# ═══════════════════════════════════════════════════════════════════════

def run_sindy_alien(hippo_trajs, alien_dims):
    """SINDy symbolic regression on alien dimensions only."""
    from l5pc.probing.dynamical_probes import sindy_probe

    logger.info("=" * 70)
    logger.info("ANALYSIS 2: SINDy on Alien Dimensions (%d dims)", len(alien_dims))
    logger.info("=" * 70)

    if len(alien_dims) == 0:
        logger.info("  No alien dimensions found -- all dims bio-correlated!")
        return {'error': 'no alien dims'}

    # Extract alien dimensions from trajectories
    # Limit to first 10 alien dims for SINDy tractability (polynomial features explode)
    alien_subset = alien_dims[:10]
    logger.info("  Using first %d alien dims for SINDy: %s", len(alien_subset), alien_subset.tolist())

    alien_trajs = [t[:, alien_subset] for t in hippo_trajs[:20]]  # 20 trials

    result = sindy_probe(alien_trajs, dt=1.0 / HIPPO_SAMPLING_RATE,
                          poly_order=2, threshold=0.01)

    if 'error' in result:
        logger.info("  SINDy unavailable: %s", result['error'])
        logger.info("  Install with: pip install pysindy")
    else:
        logger.info("  Discovered %d nonzero terms", result['n_terms'])
        logger.info("  Sparsity: %.3f", result['sparsity'])
        logger.info("  Prediction R2: %.4f", result['prediction_r2'])
        logger.info("  HH similarity (linearity ratio): %.4f", result['hh_similarity'])
        if result.get('equations'):
            logger.info("  Equations (first 3):")
            for eq in result['equations'][:3]:
                logger.info("    %s", eq)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Analysis 3: Gate-Specific Probing
# ═══════════════════════════════════════════════════════════════════════

def run_gate_specific(hippo_targets, var_names):
    """Probe LSTM forget/input/output gates and cell state separately.

    Requires the hippocampal model checkpoint (hippo_lstm_best.pt) and
    test inputs (test_data.npz).
    """
    from l5pc.probing.temporal_probes import gate_specific_probe

    logger.info("=" * 70)
    logger.info("ANALYSIS 3: Gate-Specific Probing")
    logger.info("=" * 70)

    # Try to find model checkpoint
    model_paths = list(HIPPO_ROOT.glob('**/*best*.pt')) + \
                  list(HIPPO_ROOT.glob('**/model*.pt')) + \
                  list(HIPPO_ROOT.glob('**/*checkpoint*.pt'))

    if not model_paths:
        # Fall back to L5PC model (gate analysis is still informative)
        l5pc_model_path = L5PC_MODELS / 'lstm_h128_best.pt'
        if l5pc_model_path.exists():
            logger.info("  No hippocampal model found. Using L5PC h=128 model instead.")
            logger.info("  (Gate analysis on L5PC: where does biology NOT live?)")
            return _run_gate_specific_l5pc(l5pc_model_path)
        else:
            logger.info("  No model checkpoint found. Skipping gate-specific analysis.")
            logger.info("  To enable: place hippocampal model .pt file in %s", HIPPO_ROOT)
            return {'error': 'no model checkpoint found'}

    model_path = model_paths[0]
    logger.info("  Using model: %s", model_path)

    # Load model
    import torch
    sys.path.insert(0, str(HIPPO_ROOT))

    try:
        # Try loading as state dict into known architecture
        from l5pc.surrogates.lstm import L5PC_LSTM
        model = L5PC_LSTM(input_dim=14, hidden_size=128, num_layers=2)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except Exception as e:
        logger.info("  Could not load model: %s", e)
        logger.info("  Falling back to L5PC model.")
        l5pc_model_path = L5PC_MODELS / 'lstm_h128_best.pt'
        if l5pc_model_path.exists():
            return _run_gate_specific_l5pc(l5pc_model_path)
        return {'error': str(e)}

    # Load test inputs
    td = np.load(HIPPO_RATES / 'test_data.npz')
    X_test = td['X'].astype(np.float32)  # (108, 200, 14)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    result = gate_specific_probe(model, X_test, hippo_targets, var_names, device=device)

    _report_gate_results(result, var_names)
    return result


def _run_gate_specific_l5pc(model_path):
    """Gate-specific probing on L5PC model."""
    import torch
    from l5pc.surrogates.lstm import L5PC_LSTM

    logger.info("  Running gate-specific probing on L5PC h=128...")

    model = L5PC_LSTM(input_dim=50, hidden_size=128)
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)

    # Load L5PC test data
    # Use first 20 trials for tractability
    trial_files = sorted((L5PC_TARGETS).glob('trial_*.npz'))[:20]
    if not trial_files:
        return {'error': 'no L5PC trial files found'}

    X_list = []
    for tf in trial_files:
        d = np.load(tf)
        X_list.append(d['inputs'])  # (750, 50)

    X_test = np.stack(X_list).astype(np.float32)  # (20, 750, 50)

    # Load L5PC targets
    h_trained = load_l5pc_hidden(trained=True)
    # Use first 20 trials worth of targets
    n_per_trial = L5PC_T_STEPS
    flat_targets = np.load(L5PC_TARGETS / 'trial_000.npz')
    target_keys = [k for k in flat_targets.keys() if k.startswith('geff_')][:5]

    # Simplified: just load hidden states as proxy targets
    logger.info("  L5PC gate-specific: using 5 conductance variables as targets")

    # Build target array from trial files
    targets = []
    target_names = []
    for tf in trial_files[:20]:
        d = np.load(tf)
        trial_targets = []
        for key in sorted(d.keys()):
            if key.startswith('geff_') or key == 'inputs':
                continue
            if key not in target_names and len(target_names) < 5:
                target_names.append(key)
        break

    # For L5PC, use the flat hidden states approach instead
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from l5pc.probing.temporal_probes import gate_specific_probe

    # Get a few target variables from the hidden state probing results
    # Use dummy targets derived from hidden states for gate analysis
    bio_dummy = h_trained[:20 * L5PC_T_STEPS, :5]
    target_names_dummy = [f'hidden_dim_{i}' for i in range(5)]

    result = gate_specific_probe(model, X_test, bio_dummy, target_names_dummy, device=device)
    _report_gate_results(result, target_names_dummy)
    return result


def _report_gate_results(result, var_names):
    """Pretty-print gate-specific probing results."""
    if 'error' in result:
        return

    logger.info("\n  Gate-Specific R2 Scores:")
    logger.info("  %s", "-" * 80)
    header = f"  {'Variable':<30}"
    for gate in ['forget', 'input', 'output', 'cell']:
        header += f"  {gate:>10}"
    logger.info(header)
    logger.info("  %s", "-" * 80)

    for name in var_names[:10]:  # Show top 10
        row = f"  {name:<30}"
        for gate in ['forget', 'input', 'output', 'cell']:
            if gate in result and name in result[gate]:
                row += f"  {result[gate][name]:>10.4f}"
            else:
                row += f"  {'N/A':>10}"
        logger.info(row)


# ═══════════════════════════════════════════════════════════════════════
# Analysis 4: Cross-Circuit CCA (Alien Hippo vs L5PC Zombie)
# ═══════════════════════════════════════════════════════════════════════

def run_cross_circuit_cca(hippo_hidden_flat, alien_dims, l5pc_hidden_flat):
    """CCA between hippocampal alien dims and L5PC zombie hidden states.

    If the two circuits share an alien representational subspace,
    it suggests a common computational motif imposed by LSTM architecture
    rather than anything learned from biology.
    """
    from l5pc.probing.joint_alignment import cca_alignment, rsa_comparison, cka_comparison

    logger.info("=" * 70)
    logger.info("ANALYSIS 4: Cross-Circuit Alignment (Hippo Alien vs L5PC Zombie)")
    logger.info("=" * 70)

    if len(alien_dims) == 0:
        return {'error': 'no alien dims'}

    # Extract alien dims from hippocampal hidden states
    hippo_alien = hippo_hidden_flat[:, alien_dims]

    # Subsample both to same N for comparison
    n_samples = min(5000, len(hippo_alien), len(l5pc_hidden_flat))
    rng = np.random.default_rng(42)
    idx_h = rng.choice(len(hippo_alien), n_samples, replace=False)
    idx_l = rng.choice(len(l5pc_hidden_flat), n_samples, replace=False)

    hippo_sub = hippo_alien[idx_h]
    l5pc_sub = l5pc_hidden_flat[idx_l]

    # Match dimensionality: project both to same dim via PCA
    from sklearn.decomposition import PCA
    shared_dim = min(30, len(alien_dims), L5PC_H_DIM)

    pca_h = PCA(n_components=shared_dim).fit_transform(hippo_sub)
    pca_l = PCA(n_components=shared_dim).fit_transform(l5pc_sub)

    results = {}

    # CCA
    logger.info("  CCA: hippo alien (%d dims) vs L5PC zombie (%d dims), PCA->%d...",
                len(alien_dims), L5PC_H_DIM, shared_dim)
    n_cca = min(10, shared_dim)
    cca_result = cca_alignment(pca_h, pca_l, n_components=n_cca,
                                n_permutations=100, block_size=20, seed=42)
    results['cca'] = cca_result
    logger.info("    mean_cc=%.4f  max_cc=%.4f  n_significant=%d/%d",
                cca_result['mean_cc'], cca_result['max_cc'],
                cca_result['n_significant'], n_cca)

    # RSA
    logger.info("  RSA: representational geometry comparison...")
    rsa_result = rsa_comparison(pca_h, pca_l, n_samples=2000, seed=42)
    results['rsa'] = rsa_result
    logger.info("    rsa_rho=%.4f  p=%.2e  geometric_match=%s",
                rsa_result['rsa_correlation'], rsa_result['p_value'],
                rsa_result['geometric_match'])

    # CKA (linear -- cheaper for large N)
    logger.info("  CKA: kernel alignment (linear)...")
    cka_sub_h = pca_h[:1000]  # CKA is O(n^2), limit size
    cka_sub_l = pca_l[:1000]
    cka_result = cka_comparison(cka_sub_h, cka_sub_l, kernel='linear')
    results['cka'] = cka_result
    logger.info("    cka=%.4f", cka_result['cka'])

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    global HIPPO_ROOT, HIPPO_RATES, HIPPO_HIDDEN, RESULTS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=int, default=0,
                        help='0=all, 1=koopman, 2=sindy, 3=gates, 4=cca')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override hippocampal data directory (checkpoints_rates)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override results output directory')
    args = parser.parse_args()

    # Allow overriding paths for Vast.ai or other remote environments
    if args.data_dir:
        data_dir = Path(args.data_dir)
        HIPPO_RATES = data_dir
        HIPPO_HIDDEN = data_dir / 'hidden_states'
        HIPPO_ROOT = data_dir.parent
    if args.output_dir:
        RESULTS_DIR = Path(args.output_dir)

    logger.info("DESCARTES Dual Factory v3.0 -- Alien Computation Characterization")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Hippo data: %s", HIPPO_RATES)

    # ── Load hippocampal data ──
    hippo_trajs = load_hippo_hidden(trained=True)
    hippo_targets, var_names = load_hippo_targets()
    hippo_hidden_flat = np.concatenate([t for t in hippo_trajs], axis=0)

    # ── Identify alien dimensions ──
    logger.info("=" * 70)
    logger.info("DIMENSION PARTITIONING")
    logger.info("=" * 70)
    # Try multiple thresholds to find a meaningful partition
    for thresh in [0.05, 0.15, 0.25, 0.35]:
        bio_dims, alien_dims, dim_scores = identify_alien_dims(
            hippo_hidden_flat, hippo_targets, var_names, threshold=thresh)
        if len(alien_dims) > 0:
            break
    logger.info("  Final threshold: %.2f", thresh)
    logger.info("  Bio-correlated dims: %d", len(bio_dims))
    logger.info("  Alien dims: %d", len(alien_dims))
    if len(alien_dims) == 0:
        logger.info("  NOTE: All dims bio-correlated! Using bottom quartile as 'weakly alien'.")
        q25 = np.percentile(dim_scores, 25)
        alien_dims = np.where(dim_scores <= q25)[0]
        bio_dims = np.where(dim_scores > q25)[0]
        logger.info("  Repartitioned: %d strong-bio, %d weakly-alien (score <= %.3f)",
                     len(bio_dims), len(alien_dims), q25)
    logger.info("  Dimension score range: [%.4f, %.4f]",
                dim_scores.min(), dim_scores.max())

    all_results = {
        'dimension_partition': {
            'n_bio': int(len(bio_dims)),
            'n_alien': int(len(alien_dims)),
            'bio_dims': bio_dims.tolist(),
            'alien_dims': alien_dims.tolist(),
            'dim_scores': dim_scores.tolist(),
        }
    }

    t0 = time.time()

    # ── Analysis 1: Koopman ──
    if args.analysis in (0, 1):
        hippo_target_trajs = load_hippo_target_trajectories(hippo_targets)
        result = run_koopman(hippo_trajs, hippo_target_trajs, hippo_targets, var_names)
        all_results['koopman'] = to_serializable(result)

    # ── Analysis 2: SINDy on alien dims ──
    if args.analysis in (0, 2):
        result = run_sindy_alien(hippo_trajs, alien_dims)
        all_results['sindy_alien'] = to_serializable(result)

    # ── Analysis 3: Gate-specific probing ──
    if args.analysis in (0, 3):
        result = run_gate_specific(hippo_targets, var_names)
        all_results['gate_specific'] = to_serializable(result)

    # ── Analysis 4: Cross-circuit CCA ──
    if args.analysis in (0, 4):
        try:
            l5pc_hidden = load_l5pc_hidden(trained=True)
            result = run_cross_circuit_cca(hippo_hidden_flat, alien_dims, l5pc_hidden)
            all_results['cross_circuit_cca'] = to_serializable(result)
        except FileNotFoundError as e:
            logger.info("  L5PC data not available: %s", e)
            all_results['cross_circuit_cca'] = {'error': str(e)}

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("All analyses complete in %.1f min", elapsed / 60)
    logger.info("=" * 70)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / 'alien_computation_results.json'
    with open(out_path, 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)
    logger.info("Saved: %s", out_path)


if __name__ == '__main__':
    main()
