#!/usr/bin/env python3
"""
DESCARTES -- Real Resample Ablation on 5 Nonlinear Variables

Uses the actual HippocampalLSTM forward pass with resample-clamping
at every timestep, then evaluates Ridge probes on the ablated hidden states.

This is the real causal test: does ablating k hidden dims *during recurrent
dynamics* degrade decoding of each nonlinear variable more than chance?

Model architecture: 2-layer GateAccessLSTM, h=128, concat hidden -> 256 dims.
Resample-clamp operates on LAST layer (dims 128-255 of the 256-d hidden).

Requires:
  - checkpoints_rates/lstm_best.pt (or sweep_h128/lstm_best.pt)
  - checkpoints_rates/test_data.npz
  - checkpoints_rates/probe_targets.npy
  - checkpoints_rates/probe_variable_names.json
  - checkpoints_rates/hidden_states/lstm_trained.npy

Usage:
  python scripts/run_real_resample_ablation.py \
    --data-dir /path/to/checkpoints_rates \
    --output-dir results/real_resample_ablation
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NONLINEAR_VARS = ["I_h_CA1", "I_KCa_CA1", "Ca_i_CA1", "g_NMDA_SC", "m_h_CA1"]
K_VALUES = [10, 25, 50, 75, 100]
N_RANDOM_REPS = 10
BATCH_SIZE = 32
HIDDEN_SIZE = 128  # per layer; 256 total (2 layers concatenated)


# ── Model definition (from hippocampal_mimo) ─────────────────────

class GateAccessLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x, hc):
        h_prev, c_prev = hc
        gates = self.W_x(x) + self.W_h(h_prev)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class HippocampalLSTM(nn.Module):
    def __init__(self, input_dim=14, hidden_size=128, num_layers=2,
                 output_dim=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_sz = input_dim if layer == 0 else hidden_size
            self.cells.append(GateAccessLSTMCell(in_sz, hidden_size))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward_with_resample_collect(self, x, clamp_dims=None,
                                       empirical_pool=None):
        """
        Forward pass with resample-clamping on last layer hidden state.
        Returns both model output AND concatenated hidden states (256-d).

        clamp_dims: list of dim indices in [0, hidden_size) for last layer
        empirical_pool: dict mapping dim_index -> np.array of empirical values
        """
        batch, seq_len, _ = x.shape
        device = x.device
        h = [torch.zeros(batch, self.hidden_size, device=device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch, self.hidden_size, device=device)
             for _ in range(self.num_layers)]
        outs = []
        all_hidden = []

        for t in range(seq_len):
            inp = x[:, t, :]
            for L in range(self.num_layers):
                h_new, c_new = self.cells[L](inp, (h[L], c[L]))
                h[L] = h_new
                c[L] = c_new
                inp = h_new
                if self.dropout and L < self.num_layers - 1:
                    inp = self.dropout(inp)

            # Resample-clamp on last layer
            if clamp_dims is not None and empirical_pool is not None:
                h[-1] = h[-1].clone()
                for dim_idx in clamp_dims:
                    pool = empirical_pool[dim_idx]
                    samples = pool[np.random.randint(0, len(pool), size=batch)]
                    h[-1][:, dim_idx] = torch.tensor(
                        samples, dtype=torch.float32, device=device
                    )
                inp = h[-1]

            outs.append(torch.sigmoid(self.output_proj(inp)))
            # Collect concatenated hidden from all layers
            h_cat = torch.cat(h, dim=-1).detach()
            all_hidden.append(h_cat)

        output = torch.stack(outs, dim=1)
        hidden = torch.stack(all_hidden, dim=1)  # (batch, seq_len, 256)
        return output, hidden

    def set_inference_mode(self):
        self.training = False
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.training = False


# ── Helpers ──────────────────────────────────────────────────────

def batched_forward_collect(model, X, clamp_dims, empirical_pool, device):
    """Run forward pass in batches, collecting hidden states."""
    model.set_inference_mode()
    all_outputs = []
    all_hidden = []
    with torch.no_grad():
        for i in range(0, X.shape[0], BATCH_SIZE):
            batch = torch.tensor(
                X[i:i + BATCH_SIZE], dtype=torch.float32, device=device
            )
            out, hid = model.forward_with_resample_collect(
                batch, clamp_dims, empirical_pool
            )
            all_outputs.append(out.cpu().numpy())
            all_hidden.append(hid.cpu().numpy())
    outputs = np.concatenate(all_outputs, axis=0)
    hidden = np.concatenate(all_hidden, axis=0)
    return outputs, hidden


def flatten_hidden(hidden_3d):
    """(batch, seq_len, 256) -> (batch * seq_len, 256)"""
    b, t, d = hidden_3d.shape
    return hidden_3d.reshape(b * t, d)


def train_ridge_probes(hidden_flat, targets, var_names, target_vars):
    """Train Ridge probes for each target variable. Returns dict of fitted models."""
    probes = {}
    for var in target_vars:
        if var not in var_names:
            logger.warning(f"Variable '{var}' not in var_names, skipping")
            continue
        vidx = var_names.index(var)
        y = targets[:, vidx]

        # Match lengths
        min_len = min(len(hidden_flat), len(y))
        ridge = Ridge(alpha=1.0)
        ridge.fit(hidden_flat[:min_len], y[:min_len])

        # Cross-validated R2 for baseline
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(ridge, hidden_flat[:min_len], y[:min_len], cv=kf)
        baseline_r2 = float(np.mean(scores))

        probes[var] = {
            "model": ridge,
            "var_idx": vidx,
            "baseline_r2": baseline_r2,
        }
        logger.info(f"  Ridge probe for {var}: baseline R2 = {baseline_r2:.4f}")

    return probes


def evaluate_probe(probe, hidden_flat, targets, var_idx):
    """Evaluate a fitted Ridge probe on (possibly ablated) hidden states."""
    y = targets[:, var_idx]
    min_len = min(len(hidden_flat), len(y))
    pred = probe.predict(hidden_flat[:min_len])
    y_sub = y[:min_len]
    ss_res = np.sum((y_sub - pred) ** 2)
    ss_tot = np.sum((y_sub - y_sub.mean()) ** 2) + 1e-10
    return 1 - ss_res / ss_tot


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real resample ablation on 5 nonlinear variables"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Path to checkpoints_rates directory")
    parser.add_argument("--output-dir", default="results/real_resample_ablation")
    parser.add_argument("--device", default=None,
                        help="Device (auto-detect if not specified)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    T0 = time.time()

    # ── Load model ──────────────────────────────────────────────
    ckpt_path = data_dir / "lstm_best.pt"
    if not ckpt_path.exists():
        ckpt_path = data_dir / "sweep_h128" / "lstm_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {data_dir}")

    model = HippocampalLSTM(hidden_size=HIDDEN_SIZE)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.to(device)
    model.set_inference_mode()
    logger.info(f"Model loaded from {ckpt_path}")

    # ── Load data ───────────────────────────────────────────────
    test_data = np.load(str(data_dir / "test_data.npz"))
    X_test = test_data["X"]
    Y_test = test_data["Y"]
    logger.info(f"Test data: X={X_test.shape}, Y={Y_test.shape}")

    targets = np.load(str(data_dir / "probe_targets.npy"))
    with open(data_dir / "probe_variable_names.json") as f:
        var_names = json.load(f)
    logger.info(f"Probe targets: {targets.shape}, {len(var_names)} variables")

    # Load pre-extracted hidden states for empirical pools + probe training
    hidden_path = data_dir / "hidden_states" / "lstm_trained.npy"
    if not hidden_path.exists():
        hidden_path = data_dir / "sweep_h128" / "trained_hidden.npy"
    trained_hidden = np.load(str(hidden_path))
    logger.info(f"Pre-extracted hidden states: {trained_hidden.shape}")

    # Last layer hidden = dims [128:256]
    last_hidden = trained_hidden[:, HIDDEN_SIZE:]
    assert last_hidden.shape[1] == HIDDEN_SIZE

    # ── Build empirical pools for resample-clamping ─────────────
    empirical_pools = {}
    for d in range(HIDDEN_SIZE):
        empirical_pools[d] = last_hidden[:, d].copy()

    # ── Compute per-dim correlations with each nonlinear var ────
    logger.info("\nPer-variable dim correlations (last layer):")
    var_dim_corrs = {}
    for var in NONLINEAR_VARS:
        if var not in var_names:
            continue
        vidx = var_names.index(var)
        y = targets[:, vidx]
        min_len = min(len(last_hidden), len(y))
        corrs = np.array([
            np.corrcoef(last_hidden[:min_len, d], y[:min_len])[0, 1]
            if np.std(last_hidden[:min_len, d]) > 1e-10 else 0.0
            for d in range(HIDDEN_SIZE)
        ])
        corrs = np.nan_to_num(corrs)
        var_dim_corrs[var] = corrs
        ranked = np.argsort(-np.abs(corrs))
        logger.info(
            f"  {var}: max|r|={np.abs(corrs).max():.4f}, "
            f"mean|r|={np.abs(corrs).mean():.4f}, "
            f">0.3: {(np.abs(corrs) > 0.3).sum()}/{HIDDEN_SIZE}"
        )

    # ── Train Ridge probes on UNABLATED hidden states ───────────
    logger.info("\nTraining Ridge probes on unablated hidden states...")
    probes = train_ridge_probes(trained_hidden, targets, var_names, NONLINEAR_VARS)

    # ── Baseline forward pass ───────────────────────────────────
    logger.info("\nBaseline forward pass...")
    base_out, base_hidden = batched_forward_collect(
        model, X_test, None, None, device
    )
    base_hidden_flat = flatten_hidden(base_hidden)
    logger.info(f"Baseline hidden: {base_hidden.shape} -> flat {base_hidden_flat.shape}")

    # Verify baseline probe R2 on forward-pass hidden states
    for var, probe_info in probes.items():
        r2 = evaluate_probe(
            probe_info["model"], base_hidden_flat, targets, probe_info["var_idx"]
        )
        logger.info(f"  Baseline forward-pass R2 for {var}: {r2:.4f}")

    # ── Resample ablation per variable ──────────────────────────
    all_results = {}

    for var in NONLINEAR_VARS:
        if var not in probes:
            continue

        probe_info = probes[var]
        corrs = var_dim_corrs.get(var)
        if corrs is None:
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"  RESAMPLE ABLATION: {var}")
        logger.info(f"{'='*70}")

        # Rank dims by correlation with THIS variable
        ranked = np.argsort(-np.abs(corrs))

        var_results = {
            "baseline_r2": probe_info["baseline_r2"],
            "k_results": {},
        }

        for k in K_VALUES:
            if k > HIDDEN_SIZE:
                continue

            t0 = time.time()
            top_k = ranked[:k].tolist()
            min_abs_r = float(np.abs(corrs[ranked[k - 1]]))

            # --- Target ablation: clamp top-k dims for THIS variable ---
            target_pool = {d: empirical_pools[d] for d in top_k}
            _, abl_hidden = batched_forward_collect(
                model, X_test, top_k, target_pool, device
            )
            abl_flat = flatten_hidden(abl_hidden)
            target_r2 = evaluate_probe(
                probe_info["model"], abl_flat, targets, probe_info["var_idx"]
            )
            target_degradation = probe_info["baseline_r2"] - target_r2

            # --- Random control: clamp k random dims ---
            rng = np.random.default_rng(42)
            random_r2s = []
            for _ in range(N_RANDOM_REPS):
                rand_dims = rng.choice(HIDDEN_SIZE, k, replace=False).tolist()
                rand_pool = {d: empirical_pools[d] for d in rand_dims}
                _, rand_hidden = batched_forward_collect(
                    model, X_test, rand_dims, rand_pool, device
                )
                rand_flat = flatten_hidden(rand_hidden)
                rand_r2 = evaluate_probe(
                    probe_info["model"], rand_flat, targets, probe_info["var_idx"]
                )
                random_r2s.append(rand_r2)

            rand_mean = np.mean(random_r2s)
            rand_std = np.std(random_r2s)
            z_score = (target_r2 - rand_mean) / rand_std if rand_std > 1e-10 else 0.0
            verdict = "CAUSAL" if z_score < -2 else "BYPRODUCT"

            elapsed = time.time() - t0
            logger.info(
                f"  k={k:>3d} ({100*k/HIDDEN_SIZE:.0f}%): "
                f"R2={target_r2:.4f} (deg={target_degradation:.4f}), "
                f"rand={rand_mean:.4f}+/-{rand_std:.4f}, "
                f"z={z_score:+.2f} -> {verdict}  ({elapsed:.1f}s)"
            )

            var_results["k_results"][str(k)] = {
                "pct_dims": round(100 * k / HIDDEN_SIZE, 1),
                "min_abs_corr_clamped": round(min_abs_r, 4),
                "target_r2": round(target_r2, 6),
                "target_degradation": round(target_degradation, 6),
                "random_mean_r2": round(float(rand_mean), 6),
                "random_std_r2": round(float(rand_std), 6),
                "z_score": round(float(z_score), 2),
                "verdict": verdict,
            }

        all_results[var] = var_results

    # ── Cross-variable analysis ─────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(f"  CROSS-VARIABLE SUMMARY")
    logger.info(f"{'='*70}")

    # Check dim overlap between variable-specific ablation sets
    overlap_analysis = {}
    for i, v1 in enumerate(NONLINEAR_VARS):
        if v1 not in var_dim_corrs:
            continue
        r1 = np.argsort(-np.abs(var_dim_corrs[v1]))[:50]
        for v2 in NONLINEAR_VARS[i + 1:]:
            if v2 not in var_dim_corrs:
                continue
            r2 = np.argsort(-np.abs(var_dim_corrs[v2]))[:50]
            shared = len(set(r1) & set(r2))
            jaccard = shared / len(set(r1) | set(r2))
            pair = f"{v1}_vs_{v2}"
            overlap_analysis[pair] = {
                "shared_top50": shared,
                "jaccard": round(jaccard, 3),
            }
            logger.info(f"  {v1} vs {v2}: {shared}/50 shared top dims, Jaccard={jaccard:.3f}")

    # Summary table
    logger.info(f"\n{'='*70}")
    logger.info(f"  VERDICT TABLE")
    logger.info(f"{'='*70}")
    header = f"  {'Variable':<15}" + "".join(f"{'k='+str(k):>10}" for k in K_VALUES)
    logger.info(header)
    logger.info(f"  {'-'*65}")
    for var, res in all_results.items():
        row = f"  {var:<15}"
        for k in K_VALUES:
            ks = str(k)
            if ks in res["k_results"]:
                v = res["k_results"][ks]["verdict"]
                z = res["k_results"][ks]["z_score"]
                row += f"  {v}({z:+.1f})"
            else:
                row += f"{'N/A':>10}"
        logger.info(row)

    # Mandatory pair test
    logger.info(f"\n  PAIR ANALYSIS:")
    pair_results = {}
    for pair_name, pair_vars in [
        ("Ca_cluster", ["I_KCa_CA1", "Ca_i_CA1"]),
        ("Ih_cluster", ["I_h_CA1", "m_h_CA1"]),
    ]:
        causal_counts = []
        for var in pair_vars:
            if var in all_results:
                n_causal = sum(
                    1 for kr in all_results[var]["k_results"].values()
                    if kr["verdict"] == "CAUSAL"
                )
                causal_counts.append(n_causal)
        if causal_counts:
            both_causal = all(c > 0 for c in causal_counts)
            pair_results[pair_name] = {
                "variables": pair_vars,
                "causal_counts": causal_counts,
                "both_causal": both_causal,
                "interpretation": (
                    "MANDATORY PAIR: both members causally required"
                    if both_causal
                    else "NOT mandatory: at least one member is byproduct"
                ),
            }
            logger.info(f"  {pair_name}: {'MANDATORY' if both_causal else 'NOT mandatory'} "
                        f"(causal counts: {dict(zip(pair_vars, causal_counts))})")

    # Save everything
    final = {
        "per_variable": all_results,
        "dim_overlap": overlap_analysis,
        "pair_analysis": pair_results,
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "k_values": K_VALUES,
            "n_random_reps": N_RANDOM_REPS,
            "batch_size": BATCH_SIZE,
        },
        "total_time_seconds": round(time.time() - T0, 1),
    }

    out_path = Path(args.output_dir) / "real_resample_ablation.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    logger.info(f"\nSaved to {out_path}")
    logger.info(f"Total time: {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
