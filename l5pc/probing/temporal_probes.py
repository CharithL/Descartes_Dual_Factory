"""
L5PC DESCARTES -- Tier 7: Temporal & Structural Probes

Temporal windowed probing, temporal generalization matrices,
and gate-specific LSTM probing (forget/input/output/cell).

Tests whether encoding is distributed across time or localized
to specific LSTM gating mechanisms.
"""

import logging

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 11.1  Temporal Probes
# ──────────────────────────────────────────────────────────────

def temporal_probe(hidden_trajectories, target_trajectories,
                    window_sizes=[1, 5, 10, 20, 50]):
    """
    h(t-k:t) -> bio(t): test whether encoding is distributed across time.

    If window=1 fails but window=20 succeeds, the variable is
    temporally distributed -- not absent, just not snapshot-decodable.
    """
    results = {}

    for w in window_sizes:
        X_windows = []
        y_values = []

        for h, t in zip(hidden_trajectories, target_trajectories):
            T = len(h)
            for i in range(w, T):
                X_windows.append(h[i - w:i + 1].ravel())
                y_values.append(t[i])

        X = np.array(X_windows)
        y = np.array(y_values)

        ridge = Ridge(alpha=1.0)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(ridge, X, y, cv=kf)

        results[f'window_{w}'] = float(np.mean(scores))

    return results


# ──────────────────────────────────────────────────────────────
# 11.2  Temporal Generalization Matrix
# ──────────────────────────────────────────────────────────────

def temporal_generalization(hidden_trajectory, target_trajectory,
                              step=10):
    """
    Train decoder at time t, test at time t'.

    Diagonal = same-time decoding.
    Off-diagonal = cross-time generalization.

    Stable encoding -> broad off-diagonal band.
    Transient encoding -> narrow diagonal.
    """
    T = len(hidden_trajectory)
    timepoints = list(range(0, T, step))
    n_points = len(timepoints)

    gen_matrix = np.zeros((n_points, n_points))

    for i, t_train in enumerate(timepoints):
        # Train at t_train (use a small window around it)
        window = 50
        train_start = max(0, t_train - window)
        train_end = min(T, t_train + window)

        ridge = Ridge(alpha=1.0)
        ridge.fit(hidden_trajectory[train_start:train_end],
                   target_trajectory[train_start:train_end])

        for j, t_test in enumerate(timepoints):
            test_start = max(0, t_test - window)
            test_end = min(T, t_test + window)

            gen_matrix[i, j] = ridge.score(
                hidden_trajectory[test_start:test_end],
                target_trajectory[test_start:test_end])

    return {
        'generalization_matrix': gen_matrix,
        'diagonal_mean': float(np.diag(gen_matrix).mean()),
        'off_diagonal_mean': float(gen_matrix[~np.eye(n_points, dtype=bool)].mean()),
        'temporal_stability': float(np.diag(gen_matrix).mean() -
                                     gen_matrix[~np.eye(n_points, dtype=bool)].mean())
    }


# ──────────────────────────────────────────────────────────────
# 11.3  Gate-Specific LSTM Probing
# ──────────────────────────────────────────────────────────────

def gate_specific_probe(model, X_test, bio_targets, target_names,
                          device='cpu'):
    """
    Probe LSTM forget, input, output gates and cell state separately.

    The forget gate controls what information persists.
    The input gate controls what new information enters.
    If biology lives in the forget gate but not the hidden state,
    the LSTM is using biology for memory management, not output generation.
    """
    model.eval()
    X = torch.tensor(X_test, dtype=torch.float32, device=device)

    # Step-by-step LSTM forward to extract gates
    lstm = model.lstm  # Assumes model.lstm is the LSTM layer
    h_dim = lstm.hidden_size

    with torch.no_grad():
        batch_size = X.shape[0]
        h = torch.zeros(1, batch_size, h_dim, device=device)
        c = torch.zeros(1, batch_size, h_dim, device=device)

        forget_gates = []
        input_gates = []
        output_gates = []
        cell_states = []

        for t in range(X.shape[1]):
            x_t = X[:, t:t + 1, :]
            _, (h, c) = lstm(x_t, (h, c))

            # Extract gates from LSTM internals
            # For standard PyTorch LSTM, gates are not directly exposed.
            # Reconstruct from weights:
            W_ii, W_if, W_ig, W_io = lstm.weight_ih_l0.chunk(4, 0)
            W_hi, W_hf, W_hg, W_ho = lstm.weight_hh_l0.chunk(4, 0)
            b_ii, b_if, b_ig, b_io = lstm.bias_ih_l0.chunk(4, 0)
            b_hi, b_hf, b_hg, b_ho = lstm.bias_hh_l0.chunk(4, 0)

            x_flat = X[:, t, :]
            h_prev = h.squeeze(0)

            i_gate = torch.sigmoid(x_flat @ W_ii.T + b_ii + h_prev @ W_hi.T + b_hi)
            f_gate = torch.sigmoid(x_flat @ W_if.T + b_if + h_prev @ W_hf.T + b_hf)
            o_gate = torch.sigmoid(x_flat @ W_io.T + b_io + h_prev @ W_ho.T + b_ho)

            forget_gates.append(f_gate.cpu().numpy())
            input_gates.append(i_gate.cpu().numpy())
            output_gates.append(o_gate.cpu().numpy())
            cell_states.append(c.squeeze(0).cpu().numpy())

    # Stack and probe each gate type
    results = {}
    gate_data = {
        'forget': np.stack(forget_gates, axis=1),    # (batch, time, h_dim)
        'input': np.stack(input_gates, axis=1),
        'output': np.stack(output_gates, axis=1),
        'cell': np.stack(cell_states, axis=1),
    }

    for gate_name, gate_vals in gate_data.items():
        flat_gate = gate_vals.reshape(-1, gate_vals.shape[-1])

        gate_results = {}
        for j, name in enumerate(target_names):
            target = bio_targets[:, j] if bio_targets.ndim > 1 else bio_targets
            if len(target) != len(flat_gate):
                target = np.tile(target, len(flat_gate) // len(target) + 1)[:len(flat_gate)]

            scores = cross_val_score(Ridge(1.0), flat_gate, target, cv=5)
            gate_results[name] = float(np.mean(scores))

        results[gate_name] = gate_results

    return results
