#!/usr/bin/env python3
"""
Diagnose why the inverted factory probe gives negative gamma_r2
for a vanilla LSTM that achieves CC=0.589.

Tests:
1. Pre-extracted hidden states (the validated Phase 1 data)
2. Newly-trained model hidden states
3. Raw correlation check (no Ridge, just Pearson r per dim)
4. Data alignment check
5. Ridge at multiple alpha values
6. Manual train/test Ridge (bypass cross_val_score)
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class SimpleLSTM(nn.Module):
    def __init__(self, n_in, n_out, h=128, nl=2):
        super().__init__()
        self.lstm = nn.LSTM(n_in, h, nl, batch_first=True)
        self.fc = nn.Linear(h, n_out)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o), o


def run_ridge_probe(hidden_flat, target, label=""):
    """Run Ridge probe at multiple alphas and report."""
    scaler = StandardScaler()
    h_z = scaler.fit_transform(hidden_flat)

    print(f"  Ridge probe {label}:")
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        r2 = float(np.mean(cross_val_score(
            Ridge(alpha), h_z, target, cv=5, scoring='r2')))
        status = 'PASS' if r2 > 0 else 'FAIL'
        print(f"    alpha={alpha:>7.1f}: R2={r2:+.4f} [{status}]")

    # Manual split check
    n = len(target)
    n_tr = int(0.8 * n)
    ridge = Ridge(alpha=1.0)
    ridge.fit(h_z[:n_tr], target[:n_tr])
    pred = ridge.predict(h_z[n_tr:])
    g_te = target[n_tr:]
    ss_res = np.sum((g_te - pred) ** 2)
    ss_tot = np.sum((g_te - g_te.mean()) ** 2)
    r2_man = 1.0 - ss_res / ss_tot
    print(f"    Manual split R2 = {r2_man:+.4f} "
          f"(SS_res={ss_res:.1f} SS_tot={ss_tot:.1f})")
    print(f"    pred: mean={pred.mean():.4f} std={pred.std():.4f}")
    print(f"    true: mean={g_te.mean():.4f} std={g_te.std():.4f}")
    return r2_man


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = args.device

    # Load probe targets
    probes = np.load(data_dir / 'probe_targets.npy')
    with open(data_dir / 'probe_variable_names.json') as f:
        vnames = json.load(f)
    gi = vnames.index('gamma_amp')
    gamma = probes[:, gi]

    print("=" * 60)
    print("PROBE DIAGNOSTIC")
    print("=" * 60)
    print(f"Probe targets: {probes.shape}, {len(vnames)} variables")
    print(f"gamma_amp (idx {gi}): mean={gamma.mean():.4f} "
          f"std={gamma.std():.4f} min={gamma.min():.4f} "
          f"max={gamma.max():.4f}")
    print(f"  NaN: {np.isnan(gamma).sum()}, "
          f"Inf: {np.isinf(gamma).sum()}, "
          f"Constant: {gamma.std() < 1e-10}")
    print()

    # Load test data
    td = np.load(data_dir / 'test_data.npz')
    X, Y = td['X'], td['Y']
    n_trials, n_steps, n_in = X.shape
    n_out = Y.shape[2]
    n_samples = n_trials * n_steps

    print(f"test_data.npz: X={X.shape} Y={Y.shape}")
    print(f"  {n_trials} trials x {n_steps} steps = {n_samples} samples")
    print(f"  probe_targets has {probes.shape[0]} samples")
    if n_samples != probes.shape[0]:
        print(f"  *** CRITICAL MISMATCH: {n_samples} != "
              f"{probes.shape[0]} ***")
    print()

    # ---- TEST 1: Pre-extracted hidden states ----
    print("-" * 60)
    print("TEST 1: Pre-extracted hidden states (Phase 1 validated)")
    print("-" * 60)
    h_path = data_dir / 'hidden_states' / 'lstm_trained.npy'
    if h_path.exists():
        h_pre = np.load(h_path)
        print(f"  Shape: {h_pre.shape}")
        print(f"  mean={h_pre.mean():.4f} std={h_pre.std():.4f}")
        print(f"  NaN: {np.isnan(h_pre).sum()}")

        # Raw correlations
        n_dims = min(h_pre.shape[1], 256)
        corrs = [abs(np.corrcoef(h_pre[:, d], gamma)[0, 1])
                 for d in range(n_dims)]
        print(f"  Raw max|r| (first {n_dims} dims): "
              f"{max(corrs):.4f}")
        print(f"  Dims with |r| > 0.3: "
              f"{sum(1 for c in corrs if c > 0.3)}/{n_dims}")

        run_ridge_probe(h_pre, gamma, "(pre-extracted)")
    else:
        print(f"  SKIPPED: {h_path} not found")
    print()

    # ---- TEST 2: New model hidden states ----
    print("-" * 60)
    print("TEST 2: Newly-trained LSTM h=128")
    print("-" * 60)

    model = SimpleLSTM(n_in, n_out, h=128, nl=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    Yt = torch.tensor(Y, dtype=torch.float32).to(device)

    for ep in range(150):
        model.train()
        pred, _ = model(Xt)
        loss = crit(pred, Yt)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred, hidden = model(Xt)

    cc = float(np.corrcoef(
        pred.cpu().numpy().ravel(), Y.ravel())[0, 1])
    print(f"  Output CC = {cc:.3f}")

    h_new = hidden.cpu().numpy()  # (108, 200, 128)
    h_flat = h_new.reshape(-1, 128)
    print(f"  Hidden (3D): {h_new.shape}")
    print(f"  Hidden (2D): {h_flat.shape}")
    print(f"  mean={h_flat.mean():.4f} std={h_flat.std():.4f}")
    print(f"  NaN: {np.isnan(h_flat).sum()}")

    # Raw correlations
    corrs_new = [abs(np.corrcoef(h_flat[:, d], gamma)[0, 1])
                 for d in range(128)]
    print(f"  Raw max|r|: {max(corrs_new):.4f}")
    print(f"  Dims with |r| > 0.3: "
          f"{sum(1 for c in corrs_new if c > 0.3)}/128")
    print(f"  Top 5 dims: "
          f"{sorted(corrs_new, reverse=True)[:5]}")

    run_ridge_probe(h_flat, gamma, "(new model)")
    print()

    # ---- TEST 3: Temporal alignment check ----
    print("-" * 60)
    print("TEST 3: Temporal alignment sanity")
    print("-" * 60)
    # If probe targets are aligned with test_data trials,
    # then gamma[0:200] should correspond to trial 0,
    # gamma[200:400] to trial 1, etc.
    # Check if the OUTPUT of the model correlates with Y
    # when both are in the same order.
    pred_flat = pred.cpu().numpy().reshape(-1)
    y_flat = Y.reshape(-1)
    print(f"  pred vs Y correlation: "
          f"{np.corrcoef(pred_flat, y_flat)[0, 1]:.4f}")

    # Check first trial
    pred_t0 = pred.cpu().numpy()[0, :, 0]  # trial 0, all steps
    y_t0 = Y[0, :, 0]
    gamma_t0 = gamma[0:200]
    print(f"  Trial 0 pred vs Y: "
          f"{np.corrcoef(pred_t0, y_t0)[0, 1]:.4f}")
    print(f"  Trial 0 pred vs gamma[0:200]: "
          f"{np.corrcoef(pred_t0, gamma_t0)[0, 1]:.4f}")
    print(f"  Trial 0 Y vs gamma[0:200]: "
          f"{np.corrcoef(y_t0, gamma_t0)[0, 1]:.4f}")

    # Shuffle test: if we shuffle gamma, R2 should be near 0
    print()
    print("-" * 60)
    print("TEST 4: Shuffle control")
    print("-" * 60)
    gamma_shuffled = gamma.copy()
    np.random.shuffle(gamma_shuffled)
    scaler = StandardScaler()
    h_z = scaler.fit_transform(h_flat)
    r2_shuf = float(np.mean(cross_val_score(
        Ridge(1.0), h_z, gamma_shuffled, cv=5, scoring='r2')))
    print(f"  Ridge R2 on SHUFFLED gamma: {r2_shuf:+.4f}")
    print(f"  (Should be ~0.0)")

    print()
    print("=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
