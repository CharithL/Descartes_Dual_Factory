#!/usr/bin/env python3
"""
DESCARTES -- Gamma Amplitude Feature Analysis + Nonlinear Variable Ablation

Three analyses on hippocampal LSTM hidden states:
  Task 1: Correlation structure (gamma_amp vs all 25 biological variables)
  Task 2: Resample ablation on 5 nonlinear-encoded variables at k=[10,25,50,75,100]
  Task 3: Ablation degradation characterization (encoding overlap, concentration)

Usage:
  python scripts/run_gamma_amp_analysis.py \
    --data-dir /workspace/hippocampal_data \
    --checkpoint /workspace/hippocampal_data/best_model.pt \
    --output-dir results/gamma_amp_analysis
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# The 5 nonlinear-encoded variables identified by MLP probing
NONLINEAR_VARS = [
    "I_h_CA1",
    "I_KCa_CA1",
    "Ca_i_CA1",
    "g_NMDA_SC",
    "m_h_CA1",
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_hippocampal_data(data_dir):
    """Load hippocampal hidden states, targets, and variable names."""
    data_dir = Path(data_dir)

    # Try common file patterns
    candidates_hidden = list(data_dir.glob("*hidden*.npy")) + list(data_dir.glob("*h_states*.npy"))
    candidates_target = list(data_dir.glob("*target*.npy")) + list(data_dir.glob("*bio*.npy"))
    candidates_names = list(data_dir.glob("*names*.npy")) + list(data_dir.glob("*names*.json"))

    if not candidates_hidden:
        # Try loading from a combined npz file
        npz_files = list(data_dir.glob("*.npz"))
        if npz_files:
            data = np.load(npz_files[0], allow_pickle=True)
            keys = list(data.keys())
            logger.info(f"Found npz with keys: {keys}")
            hidden = data.get("hidden_states", data.get("h_states", data[keys[0]]))
            targets = data.get("targets", data.get("bio_targets", data[keys[1]]))
            names = data.get("variable_names", data.get("names", None))
            if names is not None:
                names = list(names)
            return hidden, targets, names

    hidden = np.load(candidates_hidden[0]) if candidates_hidden else None
    targets = np.load(candidates_target[0]) if candidates_target else None
    names = None
    if candidates_names:
        p = candidates_names[0]
        if p.suffix == ".json":
            with open(p) as f:
                names = json.load(f)
        else:
            names = list(np.load(p, allow_pickle=True))

    if hidden is None or targets is None:
        raise FileNotFoundError(
            f"Could not find hidden states and targets in {data_dir}. "
            f"Contents: {[f.name for f in data_dir.iterdir()]}"
        )

    return hidden, targets, names


def find_variable_index(names, var_name):
    """Find the index of a variable name (case-insensitive partial match)."""
    if names is None:
        return None
    var_lower = var_name.lower()
    for i, n in enumerate(names):
        if n.lower() == var_lower or var_lower in n.lower():
            return i
    return None


# ---------------------------------------------------------------------------
# Task 1: Correlation Structure
# ---------------------------------------------------------------------------

def task1_correlation_structure(hidden, targets, names, output_dir):
    """
    Compute correlation between gamma_amp and all 25 biological variables,
    plus the correlation between gamma_amp encoding dimensions and each variable.
    """
    logger.info("=" * 60)
    logger.info("TASK 1: Correlation Structure")
    logger.info("=" * 60)

    n_vars = targets.shape[1] if targets.ndim > 1 else 1
    h_flat = hidden.reshape(-1, hidden.shape[-1])

    # Find gamma_amp index
    gamma_idx = find_variable_index(names, "gamma_amp")
    if gamma_idx is None:
        # Try alternative names
        for alt in ["gamma", "gamma_amplitude", "gamma_power"]:
            gamma_idx = find_variable_index(names, alt)
            if gamma_idx is not None:
                break

    if gamma_idx is None:
        logger.warning("gamma_amp not found in variable names. Using index 0 as fallback.")
        gamma_idx = 0

    gamma_name = names[gamma_idx] if names else f"var_{gamma_idx}"
    logger.info(f"gamma_amp variable: '{gamma_name}' at index {gamma_idx}")

    # Flatten targets to match hidden
    t_flat = targets.reshape(-1, n_vars) if targets.ndim > 1 else targets.reshape(-1, 1)

    # Bio-bio correlation: gamma_amp vs each variable
    gamma_vals = t_flat[:, gamma_idx]
    bio_correlations = {}
    for j in range(n_vars):
        vname = names[j] if names else f"var_{j}"
        corr = np.corrcoef(gamma_vals, t_flat[:, j])[0, 1]
        bio_correlations[vname] = float(corr)

    # Sort by absolute correlation
    sorted_bio = sorted(bio_correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    logger.info(f"\nBio-bio correlations (gamma_amp vs each variable):")
    logger.info(f"{'Variable':<30} {'Correlation':>12}")
    logger.info("-" * 44)
    for vname, corr in sorted_bio:
        marker = " ***" if vname in NONLINEAR_VARS else ""
        logger.info(f"{vname:<30} {corr:>12.4f}{marker}")

    # Hidden-dim correlations: which hidden dims encode gamma_amp
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score, KFold

    # Per-dimension correlation with gamma_amp
    n_dims = h_flat.shape[1]
    dim_gamma_corr = np.zeros(n_dims)
    min_len = min(len(h_flat), len(gamma_vals))
    for d in range(n_dims):
        dim_gamma_corr[d] = np.corrcoef(h_flat[:min_len, d], gamma_vals[:min_len])[0, 1]

    # Ridge probe: gamma_amp from hidden states
    ridge = Ridge(alpha=1.0)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(ridge, h_flat[:min_len], gamma_vals[:min_len], cv=kf)
    gamma_r2 = float(np.mean(scores))

    logger.info(f"\nGamma_amp Ridge R2 from hidden states: {gamma_r2:.4f}")
    logger.info(f"Top 10 hidden dims correlated with gamma_amp:")
    top_dims = np.argsort(-np.abs(dim_gamma_corr))[:10]
    for d in top_dims:
        logger.info(f"  dim {d}: r = {dim_gamma_corr[d]:.4f}")

    # Cross-correlation matrix: top nonlinear vars encoding dims overlap
    encoding_dims_per_var = {}
    for var in NONLINEAR_VARS:
        vidx = find_variable_index(names, var)
        if vidx is None:
            continue
        v_flat = t_flat[:min_len, vidx]
        dim_corr = np.array([
            np.corrcoef(h_flat[:min_len, d], v_flat)[0, 1]
            for d in range(n_dims)
        ])
        # Top 20% dims by absolute correlation
        threshold = np.percentile(np.abs(dim_corr), 80)
        encoding_dims_per_var[var] = set(np.where(np.abs(dim_corr) >= threshold)[0])

    # Jaccard overlap between gamma_amp encoding dims and each nonlinear var
    gamma_top_dims = set(np.where(np.abs(dim_gamma_corr) >= np.percentile(np.abs(dim_gamma_corr), 80))[0])
    overlap_results = {}
    for var, dims in encoding_dims_per_var.items():
        jaccard = len(gamma_top_dims & dims) / max(len(gamma_top_dims | dims), 1)
        overlap_results[var] = {
            "jaccard_overlap": float(jaccard),
            "shared_dims": len(gamma_top_dims & dims),
            "gamma_dims": len(gamma_top_dims),
            "var_dims": len(dims),
        }
        logger.info(f"  gamma_amp vs {var}: Jaccard={jaccard:.3f}, shared={len(gamma_top_dims & dims)}")

    results = {
        "gamma_variable": gamma_name,
        "gamma_index": gamma_idx,
        "bio_correlations": bio_correlations,
        "gamma_r2_from_hidden": gamma_r2,
        "top_encoding_dims": {int(d): float(dim_gamma_corr[d]) for d in top_dims},
        "encoding_overlap_with_nonlinear_vars": overlap_results,
    }

    out_path = Path(output_dir) / "task1_correlation_structure.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {out_path}")

    return results


# ---------------------------------------------------------------------------
# Task 2: Resample Ablation on 5 Nonlinear Variables
# ---------------------------------------------------------------------------

def resample_ablation_from_hidden(hidden, targets, var_idx, model_output_layer,
                                   k_values, n_resamples=50, rng=None):
    """
    Resample ablation: for each k, replace k random hidden dims with
    values drawn from the marginal distribution, then measure output degradation.

    Uses the model's output layer (linear) to map hidden to predictions.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    h_flat = hidden.reshape(-1, hidden.shape[-1])
    t_flat = targets.reshape(-1) if targets.ndim > 1 else targets
    min_len = min(len(h_flat), len(t_flat))
    h_flat = h_flat[:min_len]
    t_flat = t_flat[:min_len]

    n_dims = h_flat.shape[1]

    # Baseline prediction
    if model_output_layer is not None:
        import torch
        with torch.no_grad():
            h_tensor = torch.tensor(h_flat, dtype=torch.float32)
            baseline_pred = model_output_layer(h_tensor).numpy()
            if baseline_pred.ndim > 1:
                baseline_pred = baseline_pred[:, var_idx]
    else:
        # Fallback: use Ridge as proxy
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(h_flat, t_flat)
        baseline_pred = ridge.predict(h_flat)

    baseline_r2 = 1 - np.sum((t_flat - baseline_pred) ** 2) / (
        np.sum((t_flat - t_flat.mean()) ** 2) + 1e-10
    )

    results_by_k = {}
    for k in k_values:
        if k > n_dims:
            logger.warning(f"k={k} exceeds n_dims={n_dims}, skipping")
            continue

        degradations = []
        for _ in range(n_resamples):
            # Select k random dimensions to ablate
            ablate_dims = rng.choice(n_dims, size=k, replace=False)

            # Create ablated hidden states
            h_ablated = h_flat.copy()
            for d in ablate_dims:
                # Resample from marginal distribution of this dimension
                h_ablated[:, d] = rng.choice(h_flat[:, d], size=len(h_flat), replace=True)

            # Predict with ablated hidden states
            if model_output_layer is not None:
                with torch.no_grad():
                    h_tensor = torch.tensor(h_ablated, dtype=torch.float32)
                    ablated_pred = model_output_layer(h_tensor).numpy()
                    if ablated_pred.ndim > 1:
                        ablated_pred = ablated_pred[:, var_idx]
            else:
                ablated_pred = ridge.predict(h_ablated)

            ablated_r2 = 1 - np.sum((t_flat - ablated_pred) ** 2) / (
                np.sum((t_flat - t_flat.mean()) ** 2) + 1e-10
            )

            degradations.append(baseline_r2 - ablated_r2)

        results_by_k[k] = {
            "mean_degradation": float(np.mean(degradations)),
            "std_degradation": float(np.std(degradations)),
            "median_degradation": float(np.median(degradations)),
            "max_degradation": float(np.max(degradations)),
            "fraction_k": k / n_dims,
        }

        logger.info(
            f"  k={k:>3d} ({k/n_dims:.0%} dims): "
            f"mean_degradation={np.mean(degradations):.4f} +/- {np.std(degradations):.4f}"
        )

    return {
        "baseline_r2": float(baseline_r2),
        "n_dims": n_dims,
        "results_by_k": results_by_k,
    }


def task2_resample_ablation(hidden, targets, names, checkpoint_path, output_dir):
    """
    Run resample ablation on the 5 nonlinear-encoded variables
    at k = [10, 25, 50, 75, 100].
    """
    logger.info("=" * 60)
    logger.info("TASK 2: Resample Ablation on 5 Nonlinear Variables")
    logger.info("=" * 60)

    k_values = [10, 25, 50, 75, 100]

    # Try to load model output layer
    output_layer = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            import torch
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            # Try to extract the output/readout layer
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            else:
                state_dict = ckpt.state_dict() if hasattr(ckpt, "state_dict") else None

            if state_dict is not None:
                # Look for output layer weights
                out_keys = [k for k in state_dict.keys() if "output" in k or "readout" in k or "fc" in k]
                if out_keys:
                    # Reconstruct a simple linear layer
                    weight_key = [k for k in out_keys if "weight" in k]
                    bias_key = [k for k in out_keys if "bias" in k]
                    if weight_key:
                        w = state_dict[weight_key[0]]
                        b = state_dict[bias_key[0]] if bias_key else torch.zeros(w.shape[0])
                        output_layer = torch.nn.Linear(w.shape[1], w.shape[0])
                        output_layer.weight.data = w
                        output_layer.bias.data = b
                        output_layer.eval()
                        logger.info(f"Loaded output layer: {w.shape[1]} -> {w.shape[0]}")
        except Exception as e:
            logger.warning(f"Could not load output layer from checkpoint: {e}")
            output_layer = None

    if output_layer is None:
        logger.info("Using Ridge proxy for ablation (no output layer found)")

    t_flat = targets.reshape(-1, targets.shape[-1]) if targets.ndim > 1 else targets.reshape(-1, 1)
    all_results = {}

    for var in NONLINEAR_VARS:
        vidx = find_variable_index(names, var)
        if vidx is None:
            logger.warning(f"Variable '{var}' not found in names, skipping")
            continue

        logger.info(f"\n--- {var} (index {vidx}) ---")
        var_targets = t_flat[:, vidx]

        result = resample_ablation_from_hidden(
            hidden, var_targets, vidx, output_layer,
            k_values=k_values, n_resamples=50,
        )
        all_results[var] = result

    # Summary table
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 60)
    header = f"{'Variable':<20}" + "".join(f"{'k='+str(k):>12}" for k in k_values)
    logger.info(header)
    logger.info("-" * len(header))
    for var, res in all_results.items():
        row = f"{var:<20}"
        for k in k_values:
            if k in res["results_by_k"]:
                deg = res["results_by_k"][k]["mean_degradation"]
                row += f"{deg:>12.4f}"
            else:
                row += f"{'N/A':>12}"
        logger.info(row)

    out_path = Path(output_dir) / "task2_resample_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    return all_results


# ---------------------------------------------------------------------------
# Task 3: Ablation Degradation Characterization
# ---------------------------------------------------------------------------

def task3_degradation_characterization(hidden, targets, names,
                                        task2_results, output_dir):
    """
    Characterize HOW ablation degrades encoding:
      - Encoding dimension overlap across variables
      - Concentration vs distribution of encoding
      - Degradation curve shape (linear, convex, concave)
    """
    logger.info("=" * 60)
    logger.info("TASK 3: Ablation Degradation Characterization")
    logger.info("=" * 60)

    h_flat = hidden.reshape(-1, hidden.shape[-1])
    t_flat = targets.reshape(-1, targets.shape[-1]) if targets.ndim > 1 else targets.reshape(-1, 1)
    min_len = min(len(h_flat), len(t_flat))
    h_flat = h_flat[:min_len]
    t_flat = t_flat[:min_len]
    n_dims = h_flat.shape[1]

    # Part A: Per-variable encoding dimension importance
    from sklearn.linear_model import Ridge

    importance_matrix = {}  # var -> dim -> importance
    top_dims_per_var = {}

    for var in NONLINEAR_VARS:
        vidx = find_variable_index(names, var)
        if vidx is None:
            continue

        v_flat = t_flat[:, vidx]
        ridge = Ridge(alpha=1.0)
        ridge.fit(h_flat, v_flat)

        # Importance = |coefficient| * std(dim)
        dim_stds = np.std(h_flat, axis=0)
        importance = np.abs(ridge.coef_) * dim_stds
        importance = importance / (importance.sum() + 1e-10)  # Normalize

        importance_matrix[var] = importance
        top_dims_per_var[var] = set(np.argsort(-importance)[:int(n_dims * 0.2)])

    # Part B: Pairwise encoding overlap (Jaccard)
    overlap_matrix = {}
    for i, v1 in enumerate(NONLINEAR_VARS):
        if v1 not in top_dims_per_var:
            continue
        for v2 in NONLINEAR_VARS[i + 1:]:
            if v2 not in top_dims_per_var:
                continue
            d1 = top_dims_per_var[v1]
            d2 = top_dims_per_var[v2]
            jaccard = len(d1 & d2) / max(len(d1 | d2), 1)
            pair_key = f"{v1}_vs_{v2}"
            overlap_matrix[pair_key] = {
                "jaccard": float(jaccard),
                "shared_count": len(d1 & d2),
                "union_count": len(d1 | d2),
            }
            logger.info(f"  {v1} vs {v2}: Jaccard = {jaccard:.3f} ({len(d1 & d2)} shared dims)")

    # Part C: Encoding concentration (Gini coefficient of importance)
    concentration = {}
    for var, importance in importance_matrix.items():
        sorted_imp = np.sort(importance)
        n = len(sorted_imp)
        cumulative = np.cumsum(sorted_imp)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_imp))) / (n * np.sum(sorted_imp) + 1e-10) - (n + 1) / n
        concentration[var] = {
            "gini_coefficient": float(gini),
            "top_10pct_share": float(np.sum(np.sort(importance)[-max(1, n // 10):]) / (np.sum(importance) + 1e-10)),
            "top_20pct_share": float(np.sum(np.sort(importance)[-max(1, n // 5):]) / (np.sum(importance) + 1e-10)),
            "effective_dims": float(np.exp(-np.sum(importance * np.log(importance + 1e-10)))),
        }
        logger.info(
            f"  {var}: Gini={gini:.3f}, "
            f"top10%={concentration[var]['top_10pct_share']:.3f}, "
            f"eff_dims={concentration[var]['effective_dims']:.1f}"
        )

    # Part D: Degradation curve shape from Task 2 results
    curve_analysis = {}
    for var, res in task2_results.items():
        k_vals = sorted(res["results_by_k"].keys())
        if len(k_vals) < 3:
            continue

        fractions = [res["results_by_k"][k]["fraction_k"] for k in k_vals]
        degradations = [res["results_by_k"][k]["mean_degradation"] for k in k_vals]

        # Fit linear to log(degradation) vs log(fraction) to get power law exponent
        valid = [(f, d) for f, d in zip(fractions, degradations) if d > 0 and f > 0]
        if len(valid) >= 2:
            log_f = np.log([v[0] for v in valid])
            log_d = np.log([v[1] for v in valid])
            slope = np.polyfit(log_f, log_d, 1)[0]

            # slope < 1 = concave (diminishing returns, distributed encoding)
            # slope > 1 = convex (accelerating, concentrated encoding)
            # slope ~ 1 = linear
            if slope < 0.8:
                shape = "concave_distributed"
            elif slope > 1.2:
                shape = "convex_concentrated"
            else:
                shape = "approximately_linear"

            curve_analysis[var] = {
                "power_law_exponent": float(slope),
                "curve_shape": shape,
                "interpretation": (
                    "Encoding is distributed across many dims"
                    if shape == "concave_distributed"
                    else "Encoding is concentrated in few dims"
                    if shape == "convex_concentrated"
                    else "Encoding is moderately distributed"
                ),
            }
            logger.info(f"  {var}: exponent={slope:.2f} -> {shape}")

    results = {
        "encoding_overlap": overlap_matrix,
        "encoding_concentration": concentration,
        "degradation_curve_analysis": curve_analysis,
        "summary": {
            "mean_pairwise_jaccard": float(
                np.mean([v["jaccard"] for v in overlap_matrix.values()])
            ) if overlap_matrix else 0.0,
            "mean_gini": float(
                np.mean([v["gini_coefficient"] for v in concentration.values()])
            ) if concentration else 0.0,
            "distributed_vars": [
                v for v, c in curve_analysis.items()
                if c["curve_shape"] == "concave_distributed"
            ],
            "concentrated_vars": [
                v for v, c in curve_analysis.items()
                if c["curve_shape"] == "convex_concentrated"
            ],
        },
    }

    out_path = Path(output_dir) / "task3_degradation_characterization.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gamma amplitude feature analysis")
    parser.add_argument("--data-dir", required=True, help="Directory with hippocampal data")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--output-dir", default="results/gamma_amp_analysis")
    parser.add_argument("--tasks", default="1,2,3", help="Comma-separated task numbers to run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    logger.info("Loading hippocampal data...")
    hidden, targets, names = load_hippocampal_data(args.data_dir)
    logger.info(f"Hidden shape: {hidden.shape}")
    logger.info(f"Targets shape: {targets.shape}")
    if names:
        logger.info(f"Variable names ({len(names)}): {names[:10]}...")
    else:
        logger.info("No variable names found; using indices")

    tasks = [int(t.strip()) for t in args.tasks.split(",")]

    task2_results = None

    if 1 in tasks:
        task1_correlation_structure(hidden, targets, names, args.output_dir)

    if 2 in tasks:
        task2_results = task2_resample_ablation(
            hidden, targets, names, args.checkpoint, args.output_dir
        )

    if 3 in tasks:
        if task2_results is None:
            # Try to load from file
            t2_path = Path(args.output_dir) / "task2_resample_ablation.json"
            if t2_path.exists():
                with open(t2_path) as f:
                    task2_results = json.load(f)
                logger.info("Loaded Task 2 results from file")
            else:
                logger.warning("Task 2 results not available; running Task 2 first")
                task2_results = task2_resample_ablation(
                    hidden, targets, names, args.checkpoint, args.output_dir
                )

        task3_degradation_characterization(
            hidden, targets, names, task2_results, args.output_dir
        )

    logger.info("\nAll requested tasks complete.")


if __name__ == "__main__":
    main()
