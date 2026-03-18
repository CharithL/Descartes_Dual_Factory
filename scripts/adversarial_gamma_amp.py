#!/usr/bin/env python3
"""
adversarial_gamma_amp.py

DESCARTES Dual Factory v3.0 -- Adversarial Zombie Experiment

Train hippocampal LSTM with anti-gamma_amp penalty. The network must
produce correct CA1 output while being punished for encoding gamma_amp
in its hidden states.

Three possible outcomes:
  A) Output collapses - gamma_amp is MANDATORY (strongest result)
  B) Output survives, alien computation replaces gamma_amp - REPLACEABLE
  C) Output survives but degrades - PARTIALLY MANDATORY

Sweep lambda_anti = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
to find the Pareto frontier of output quality vs gamma encoding.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger('adversarial')


# =========================================================================
# JSON serialization helper
# =========================================================================

def _convert_numpy(obj):
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {str(k): _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =========================================================================
# Model
# =========================================================================

class LSTMSurrogate(nn.Module):
    def __init__(self, n_input, n_output, hidden_dim=128,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_input, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, n_output)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.output_layer(lstm_out)
        return output, lstm_out


# =========================================================================
# Differentiable correlation loss
# =========================================================================

def batch_correlation(hidden, target):
    """
    Differentiable correlation between hidden states and
    target variable. Returns mean |r| across all hidden dims.

    hidden: (batch, time, hidden_dim)
    target: (batch, time, 1) or (batch, time)
    """
    if target.dim() == 2:
        target = target.unsqueeze(-1)

    # Flatten batch and time
    h_flat = hidden.reshape(-1, hidden.shape[-1])
    t_flat = target.reshape(-1, 1)

    # Center
    h_centered = h_flat - h_flat.mean(dim=0, keepdim=True)
    t_centered = t_flat - t_flat.mean(dim=0, keepdim=True)

    # Correlation per hidden dim
    h_std = h_centered.std(dim=0, keepdim=True) + 1e-8
    t_std = t_centered.std(dim=0, keepdim=True) + 1e-8

    correlations = (h_centered / h_std * t_centered / t_std).mean(dim=0)

    return correlations.abs().mean()


# =========================================================================
# Training
# =========================================================================

NONLINEAR_VARS = ['I_h_CA1', 'I_KCa_CA1', 'Ca_i_CA1',
                  'g_NMDA_SC', 'm_h_CA1']


def train_adversarial(data_dir, output_dir, lambda_anti=0.1,
                      hidden_dim=128, max_epochs=300,
                      patience=30, device='cuda', seed=42):
    """
    Train LSTM with anti-gamma_amp penalty.

    Args:
        lambda_anti: strength of anti-biology penalty.
            0.0 = normal training (baseline comparison)
            0.01-0.1 = mild pressure to avoid gamma_amp
            0.5-5.0 = strong pressure, may sacrifice output
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir) / f'lambda_{lambda_anti:.4f}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    test_data = np.load(data_dir / 'test_data.npz')
    X = test_data['X']
    Y = test_data['Y']

    # Load gamma_amp probe target
    probe_targets = np.load(data_dir / 'probe_targets.npy')
    with open(data_dir / 'probe_variable_names.json') as f:
        var_names = json.load(f)

    gamma_idx = var_names.index('gamma_amp')
    gamma_amp = probe_targets[:, gamma_idx]

    # Reshape gamma_amp to match (n_trials, n_timesteps)
    n_trials = X.shape[0]
    n_timesteps = X.shape[1]
    gamma_amp_3d = gamma_amp.reshape(n_trials, n_timesteps)

    # Nonlinear variable indices for post-hoc probing
    nonlinear_indices = [var_names.index(v) for v in NONLINEAR_VARS]

    # Train/test split
    n_train = int(0.8 * n_trials)
    idx = np.random.permutation(n_trials)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y[train_idx], dtype=torch.float32).to(device)
    X_test = torch.tensor(X[test_idx], dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y[test_idx], dtype=torch.float32).to(device)
    gamma_train = torch.tensor(
        gamma_amp_3d[train_idx], dtype=torch.float32).to(device)
    gamma_test = torch.tensor(
        gamma_amp_3d[test_idx], dtype=torch.float32).to(device)

    # Model
    n_input = X.shape[2]
    n_output = Y.shape[2]
    model = LSTMSurrogate(n_input, n_output, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    output_criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'output_loss': [], 'anti_bio_loss': [], 'total_loss': [],
        'val_output_loss': [], 'val_gamma_corr': [], 'val_cc': []
    }

    logger.info("Training lambda_anti=%.4f  hidden=%d  max_epochs=%d",
                lambda_anti, hidden_dim, max_epochs)

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        pred, hidden = model(X_train)

        out_loss = output_criterion(pred, Y_train)

        if lambda_anti > 0:
            anti_loss = batch_correlation(hidden, gamma_train)
        else:
            anti_loss = torch.tensor(0.0, device=device)

        total_loss = out_loss + lambda_anti * anti_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            pred_test, hidden_test = model(X_test)
            val_out_loss = output_criterion(pred_test, Y_test).item()
            val_gamma_corr = batch_correlation(
                hidden_test, gamma_test).item()
            val_cc = float(np.corrcoef(
                pred_test.cpu().numpy().ravel(),
                Y_test.cpu().numpy().ravel())[0, 1])

        history['output_loss'].append(out_loss.item())
        history['anti_bio_loss'].append(anti_loss.item())
        history['total_loss'].append(total_loss.item())
        history['val_output_loss'].append(val_out_loss)
        history['val_gamma_corr'].append(val_gamma_corr)
        history['val_cc'].append(val_cc)

        # Early stopping on OUTPUT loss only
        if val_out_loss < best_val_loss:
            best_val_loss = val_out_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'lstm_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("  Early stop epoch %d", epoch)
                break

        if (epoch + 1) % 50 == 0:
            logger.info("  Epoch %d: out=%.4f anti=%.4f cc=%.3f "
                        "gamma_r=%.3f", epoch + 1, out_loss.item(),
                        anti_loss.item(), val_cc, val_gamma_corr)

    # --- Post-hoc analysis on best model ---
    model.load_state_dict(torch.load(output_dir / 'lstm_best.pt',
                                     weights_only=True))
    model.eval()

    X_all = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_all, hidden_all = model(X_all)

    hidden_np = hidden_all.cpu().numpy()
    pred_np = pred_all.cpu().numpy()

    final_cc = float(np.corrcoef(pred_np.ravel(), Y.ravel())[0, 1])

    # Probe gamma_amp in hidden states (Ridge R2)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    hidden_flat = hidden_np.reshape(-1, hidden_dim)
    gamma_flat = gamma_amp

    ridge = Ridge(alpha=1.0)
    gamma_r2 = float(np.mean(cross_val_score(
        ridge, hidden_flat, gamma_flat, cv=5, scoring='r2')))

    # Probe all 6 mandatory variables
    mandatory_r2 = {}
    for var_name, var_idx in zip(
            ['gamma_amp'] + NONLINEAR_VARS,
            [gamma_idx] + nonlinear_indices):
        target = probe_targets[:, var_idx]
        r2 = float(np.mean(cross_val_score(
            Ridge(1.0), hidden_flat, target, cv=5, scoring='r2')))
        mandatory_r2[var_name] = r2

    # Untrained baseline for delta-R2
    model_untrained = LSTMSurrogate(
        n_input, n_output, hidden_dim).to(device)
    model_untrained.eval()
    with torch.no_grad():
        _, hidden_untrained = model_untrained(X_all)
    hidden_untrained_flat = hidden_untrained.cpu().numpy().reshape(
        -1, hidden_dim)

    mandatory_delta_r2 = {}
    for var_name, var_idx in zip(
            ['gamma_amp'] + NONLINEAR_VARS,
            [gamma_idx] + nonlinear_indices):
        target = probe_targets[:, var_idx]
        r2_trained = float(np.mean(cross_val_score(
            Ridge(1.0), hidden_flat, target, cv=5, scoring='r2')))
        r2_untrained = float(np.mean(cross_val_score(
            Ridge(1.0), hidden_untrained_flat, target, cv=5,
            scoring='r2')))
        mandatory_delta_r2[var_name] = r2_trained - r2_untrained

    # Save results
    results = {
        'lambda_anti': lambda_anti,
        'final_cc': final_cc,
        'gamma_amp_r2': gamma_r2,
        'gamma_amp_hidden_corr': float(history['val_gamma_corr'][-1]),
        'mandatory_r2': mandatory_r2,
        'mandatory_delta_r2': mandatory_delta_r2,
        'best_val_loss': best_val_loss,
        'epochs_trained': len(history['output_loss']),
    }

    logger.info("  lambda=%.4f: CC=%.3f gamma_R2=%.3f gamma_corr=%.3f",
                lambda_anti, final_cc, gamma_r2,
                history['val_gamma_corr'][-1])
    logger.info("  Mandatory variable R2:")
    for name, r2 in mandatory_r2.items():
        dr2 = mandatory_delta_r2[name]
        logger.info("    %s: R2=%.3f  dR2=%.3f", name, r2, dr2)

    with open(output_dir / 'adversarial_results.json', 'w') as f:
        json.dump(_convert_numpy(results), f, indent=2)

    np.save(output_dir / 'hidden_trained.npy', hidden_flat)
    np.save(output_dir / 'hidden_untrained.npy', hidden_untrained_flat)
    np.save(output_dir / 'training_history.npy', history)

    return results


# =========================================================================
# Lambda sweep
# =========================================================================

def sweep_lambda(data_dir, output_dir, device='cuda'):
    """
    Sweep anti-biology penalty strength.
    Find the Pareto frontier: max output CC vs min gamma encoding.
    """
    lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    all_results = []

    for lam in lambdas:
        logger.info("=" * 60)
        logger.info("  LAMBDA = %.4f", lam)
        logger.info("=" * 60)

        result = train_adversarial(
            data_dir, output_dir, lambda_anti=lam,
            device=device)
        all_results.append(result)

    # Print Pareto frontier
    logger.info("=" * 70)
    logger.info("PARETO FRONTIER: Output CC vs Gamma Encoding")
    logger.info("=" * 70)
    logger.info("%10s %8s %10s %12s %8s %9s",
                'Lambda', 'CC', 'gamma R2', 'gamma corr',
                'I_h dR2', 'Ca_i dR2')
    logger.info("-" * 70)

    for r in all_results:
        logger.info("%10.4f %8.3f %10.3f %12.3f %8.3f %9.3f",
                    r['lambda_anti'],
                    r['final_cc'],
                    r['gamma_amp_r2'],
                    r['gamma_amp_hidden_corr'],
                    r['mandatory_delta_r2'].get('I_h_CA1', 0),
                    r['mandatory_delta_r2'].get('Ca_i_CA1', 0))

    # Determine outcome
    baseline = all_results[0]
    strongest = all_results[-1]

    logger.info("=" * 70)
    logger.info("OUTCOME DETERMINATION")
    logger.info("=" * 70)

    cc_threshold = 0.7 * baseline['final_cc']
    gamma_suppressed = strongest['gamma_amp_r2'] < 0.05
    output_survived = strongest['final_cc'] > cc_threshold

    if not output_survived:
        found_working = False
        for r in reversed(all_results):
            if r['final_cc'] > cc_threshold:
                gamma_suppressed_at_working = r['gamma_amp_r2'] < 0.05
                if gamma_suppressed_at_working:
                    logger.info("OUTCOME A: gamma_amp is MANDATORY")
                    logger.info(
                        "  Output collapses when gamma_amp is "
                        "suppressed. The transformation REQUIRES "
                        "gamma oscillation amplitude. Every solver "
                        "must pass through it.")
                else:
                    logger.info(
                        "OUTCOME C: gamma_amp is PARTIALLY MANDATORY")
                    logger.info(
                        "  Output degrades but survives up to "
                        "lambda=%.4f. gamma_amp helps but is not "
                        "irreplaceable.", r['lambda_anti'])
                found_working = True
                break

        if not found_working:
            logger.info("OUTCOME A: gamma_amp is MANDATORY")
            logger.info(
                "  No lambda produces both good output AND "
                "suppressed gamma encoding.")

    elif gamma_suppressed and output_survived:
        logger.info("OUTCOME B: gamma_amp is REPLACEABLE")
        logger.info(
            "  The LSTM found an alternative pathway without "
            "gamma_amp. The mandatory finding was "
            "architecture-contingent.")

        logger.info("  What replaced gamma_amp?")
        for var, dr2 in strongest['mandatory_delta_r2'].items():
            baseline_dr2 = baseline['mandatory_delta_r2'].get(var, 0)
            change = dr2 - baseline_dr2
            if abs(change) > 0.05:
                direction = "INCREASED" if change > 0 else "DECREASED"
                logger.info("    %s: dR2 %s by %.3f (%.3f to %.3f)",
                            var, direction, change, baseline_dr2, dr2)

    else:
        logger.info("OUTCOME C: gamma_amp is PARTIALLY MANDATORY")
        logger.info(
            "  gamma_amp encoding reduced but not eliminated. "
            "Output preserved.")

    # Save sweep
    sweep_dir = Path(output_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / 'lambda_sweep_results.json', 'w') as f:
        json.dump(_convert_numpy(all_results), f, indent=2)

    return all_results


# =========================================================================
# Entry point
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Adversarial zombie experiment: can LSTM solve '
                    'CA3 to CA1 WITHOUT gamma_amp?')
    parser.add_argument('--data-dir', required=True,
                        help='Path to hippocampal checkpoints_rates dir')
    parser.add_argument('--output-dir', required=True,
                        help='Where to save adversarial results')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--single-lambda', type=float, default=None,
                        help='Run single lambda instead of full sweep')
    args = parser.parse_args()

    T0 = time.time()

    if args.single_lambda is not None:
        train_adversarial(args.data_dir, args.output_dir,
                          lambda_anti=args.single_lambda,
                          device=args.device)
    else:
        sweep_lambda(args.data_dir, args.output_dir,
                     device=args.device)

    logger.info("Total time: %.1fs", time.time() - T0)
