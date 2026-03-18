#!/usr/bin/env python3
"""
zombie_transfer_test.py

DESCARTES Dual Factory v3.0 -- L5PC Zombie Transfer Test

Can the L5PC's alien computation solve hippocampal CA3 to CA1?
This is the single most informative falsification test for
gamma_amp mandatory-ness.

5 conditions:
  C1: Frozen transfer (L5PC LSTM frozen, only projections trained)
  C2: Partial transfer (layer 0 frozen, layer 1 + projections trained)
  C3: Warm-start (L5PC init, all layers trainable)
  C4: Anti-bio warm-start (L5PC init + anti-gamma penalty)
  C5: Random baseline (normal training for comparison)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger('transfer')


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


class TransferLSTM(nn.Module):
    """LSTM with separable input/output projections for transfer."""

    def __init__(self, n_input, n_output, hidden_dim=128,
                 n_layers=2, dropout=0.1, lstm_input_dim=None):
        super().__init__()
        if lstm_input_dim is None:
            lstm_input_dim = n_input
        self.input_proj = nn.Linear(n_input, lstm_input_dim)
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout)
        self.output_proj = nn.Linear(hidden_dim, n_output)

    def forward(self, x):
        projected = self.input_proj(x)
        lstm_out, _ = self.lstm(projected)
        output = self.output_proj(lstm_out)
        return output, lstm_out


def load_l5pc_weights(model, l5pc_checkpoint_path, device='cuda'):
    """Load L5PC LSTM weights into transfer model."""
    l5pc_state = torch.load(l5pc_checkpoint_path,
                            map_location=device, weights_only=True)
    if isinstance(l5pc_state, dict) and 'model_state_dict' in l5pc_state:
        l5pc_state = l5pc_state['model_state_dict']

    logger.info("  L5PC checkpoint keys:")
    for k, v in l5pc_state.items():
        if hasattr(v, 'shape'):
            logger.info("    %s: %s", k, v.shape)

    lstm_keys = [k for k in l5pc_state.keys() if 'lstm' in k.lower()]
    if not lstm_keys:
        lstm_keys = [k for k in l5pc_state.keys()
                     if 'weight_ih' in k or 'weight_hh' in k or
                     'bias_ih' in k or 'bias_hh' in k]

    l5pc_input_dim = None
    for k in l5pc_state.keys():
        if 'weight_ih_l0' in k:
            l5pc_input_dim = l5pc_state[k].shape[1]
            break

    logger.info("  L5PC input dim: %s, keys: %d",
                l5pc_input_dim, len(lstm_keys))

    model_state = model.state_dict()
    loaded = 0
    for k_l5pc in lstm_keys:
        k_transfer = k_l5pc
        if not k_transfer.startswith('lstm.'):
            k_transfer = 'lstm.' + k_transfer
        if k_transfer in model_state:
            if l5pc_state[k_l5pc].shape == model_state[k_transfer].shape:
                model_state[k_transfer] = l5pc_state[k_l5pc]
                loaded += 1
            else:
                logger.info("  Shape mismatch: %s L5PC=%s model=%s",
                            k_transfer, l5pc_state[k_l5pc].shape,
                            model_state[k_transfer].shape)

    model.load_state_dict(model_state)
    logger.info("  Loaded %d weight tensors", loaded)
    return model, l5pc_input_dim


def freeze_layers(model, freeze_mode):
    """Freeze layers: 'all_lstm', 'layer0', or 'none'."""
    if freeze_mode == 'all_lstm':
        for param in model.lstm.parameters():
            param.requires_grad = False
        for param in model.input_proj.parameters():
            param.requires_grad = True
        for param in model.output_proj.parameters():
            param.requires_grad = True
    elif freeze_mode == 'layer0':
        for name, param in model.lstm.named_parameters():
            param.requires_grad = '_l0' not in name
        for param in model.input_proj.parameters():
            param.requires_grad = True
        for param in model.output_proj.parameters():
            param.requires_grad = True
    elif freeze_mode == 'none':
        for param in model.parameters():
            param.requires_grad = True

    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    n_total = sum(1 for p in model.parameters())
    logger.info("  Frozen: %d/%d, Trainable: %d/%d",
                n_frozen, n_total, n_total - n_frozen, n_total)


def batch_correlation(hidden, target):
    """Differentiable mean |correlation| for anti-bio loss."""
    if target.dim() == 2:
        target = target.unsqueeze(-1)
    h_flat = hidden.reshape(-1, hidden.shape[-1])
    t_flat = target.reshape(-1, 1)
    h_c = h_flat - h_flat.mean(dim=0, keepdim=True)
    t_c = t_flat - t_flat.mean(dim=0, keepdim=True)
    h_std = h_c.std(dim=0, keepdim=True) + 1e-8
    t_std = t_c.std(dim=0, keepdim=True) + 1e-8
    corrs = (h_c / h_std * t_c / t_std).mean(dim=0)
    return corrs.abs().mean()


MANDATORY_VARS = ['gamma_amp', 'I_h_CA1', 'I_KCa_CA1',
                  'Ca_i_CA1', 'g_NMDA_SC', 'm_h_CA1']


def train_condition(X_train, Y_train, X_test, Y_test,
                    gamma_train, gamma_test,
                    model, freeze_mode, lambda_anti=0.0,
                    max_epochs=300, patience=30, device='cuda'):
    """Train one transfer condition."""
    freeze_layers(model, freeze_mode)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = {'train_loss': [], 'val_cc': [], 'val_gamma_corr': []}

    for epoch in range(max_epochs):
        model.train()
        pred, hidden = model(X_train)
        out_loss = criterion(pred, Y_train)

        if lambda_anti > 0 and gamma_train is not None:
            anti_loss = batch_correlation(hidden, gamma_train)
            total_loss = out_loss + lambda_anti * anti_loss
        else:
            total_loss = out_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_test, hidden_test = model(X_test)
            val_loss = criterion(pred_test, Y_test).item()
            val_cc = float(np.corrcoef(
                pred_test.cpu().numpy().ravel(),
                Y_test.cpu().numpy().ravel())[0, 1])
            val_gamma = 0.0
            if gamma_test is not None:
                val_gamma = batch_correlation(
                    hidden_test, gamma_test).item()

        history['train_loss'].append(out_loss.item())
        history['val_cc'].append(val_cc)
        history['val_gamma_corr'].append(val_gamma)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("    Early stop epoch %d", epoch)
                break

        if (epoch + 1) % 50 == 0:
            logger.info("    Epoch %d: loss=%.4f cc=%.3f gamma=%.3f",
                        epoch + 1, out_loss.item(), val_cc, val_gamma)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def probe_mandatory(model, X_all, probe_targets, var_names,
                    hidden_dim, device):
    """Probe mandatory variables in transfer model hidden states."""
    model.eval()
    X_t = torch.tensor(X_all, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, hidden = model(X_t)
    h_flat = hidden.cpu().numpy().reshape(-1, hidden_dim)

    # Untrained baseline
    n_in = X_all.shape[2]
    lstm_in = model.input_proj.in_features
    m_rand = TransferLSTM(n_in, 1, hidden_dim,
                          lstm_input_dim=lstm_in).to(device)
    m_rand.eval()
    with torch.no_grad():
        _, h_rand = m_rand(X_t)
    h_rand_flat = h_rand.cpu().numpy().reshape(-1, hidden_dim)

    results = {}
    for vname in MANDATORY_VARS:
        if vname not in var_names:
            continue
        vidx = var_names.index(vname)
        tgt = probe_targets[:, vidx]

        r2_t = float(np.mean(cross_val_score(
            Ridge(1.0), h_flat, tgt, cv=5, scoring='r2')))
        r2_u = float(np.mean(cross_val_score(
            Ridge(1.0), h_rand_flat, tgt, cv=5, scoring='r2')))

        corrs = np.array([
            np.corrcoef(h_flat[:, d], tgt)[0, 1]
            for d in range(hidden_dim)])
        max_r = float(np.max(np.abs(corrs)))
        n_above = int(np.sum(np.abs(corrs) > 0.3))

        results[vname] = {
            'r2_trained': r2_t, 'r2_untrained': r2_u,
            'delta_r2': r2_t - r2_u,
            'max_abs_correlation': max_r,
            'n_dims_above_0.3': n_above,
        }
    return results


def run_experiment(hippo_data_dir, l5pc_checkpoint_path,
                   output_dir, hidden_dim=128, device='cuda',
                   seed=42):
    """Run all 5 transfer conditions."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    hippo_dir = Path(hippo_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading hippocampal data...")
    test_data = np.load(hippo_dir / 'test_data.npz')
    X = test_data['X']
    Y = test_data['Y']

    probe_targets = np.load(hippo_dir / 'probe_targets.npy')
    with open(hippo_dir / 'probe_variable_names.json') as f:
        var_names = json.load(f)

    n_trials, n_timesteps, n_input = X.shape
    n_output = Y.shape[2]
    logger.info("  X=%s Y=%s", X.shape, Y.shape)

    gamma_idx = var_names.index('gamma_amp')
    gamma_amp = probe_targets[:, gamma_idx].reshape(n_trials, n_timesteps)

    n_train = int(0.8 * n_trials)
    idx = np.random.permutation(n_trials)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_tr = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    Y_tr = torch.tensor(Y[train_idx], dtype=torch.float32).to(device)
    X_te = torch.tensor(X[test_idx], dtype=torch.float32).to(device)
    Y_te = torch.tensor(Y[test_idx], dtype=torch.float32).to(device)
    g_tr = torch.tensor(
        gamma_amp[train_idx], dtype=torch.float32).to(device)
    g_te = torch.tensor(
        gamma_amp[test_idx], dtype=torch.float32).to(device)

    # Get L5PC input dim
    logger.info("Loading L5PC checkpoint: %s", l5pc_checkpoint_path)
    tmp = TransferLSTM(n_input, n_output, hidden_dim,
                       lstm_input_dim=n_input).to(device)
    _, l5pc_input_dim = load_l5pc_weights(
        tmp, l5pc_checkpoint_path, device)
    if l5pc_input_dim is None:
        l5pc_input_dim = n_input
        logger.warning("  Could not get L5PC input dim, using %d",
                        n_input)
    del tmp

    conditions = {
        'C1_frozen_transfer': {
            'desc': 'L5PC LSTM frozen, only projections trained',
            'freeze': 'all_lstm', 'l5pc': True, 'lam': 0.0,
        },
        'C2_partial_transfer': {
            'desc': 'L5PC layer 0 frozen, layer 1+proj trained',
            'freeze': 'layer0', 'l5pc': True, 'lam': 0.0,
        },
        'C3_warm_start': {
            'desc': 'L5PC init, all layers trainable',
            'freeze': 'none', 'l5pc': True, 'lam': 0.0,
        },
        'C4_anti_bio_warm_start': {
            'desc': 'L5PC init + anti-gamma penalty',
            'freeze': 'none', 'l5pc': True, 'lam': 0.01,
        },
        'C5_random_baseline': {
            'desc': 'Random init, normal training',
            'freeze': 'none', 'l5pc': False, 'lam': 0.0,
        },
    }

    all_results = {}

    for cname, spec in conditions.items():
        logger.info("=" * 60)
        logger.info("  CONDITION: %s", cname)
        logger.info("  %s", spec['desc'])
        logger.info("=" * 60)

        inp_dim = l5pc_input_dim if spec['l5pc'] else n_input
        model = TransferLSTM(n_input, n_output, hidden_dim,
                             lstm_input_dim=inp_dim).to(device)

        if spec['l5pc']:
            model, _ = load_l5pc_weights(
                model, l5pc_checkpoint_path, device)

        gtr = g_tr if spec['lam'] > 0 else None
        gte = g_te if spec['lam'] > 0 else None

        model, history = train_condition(
            X_tr, Y_tr, X_te, Y_te, gtr, gte,
            model, spec['freeze'], lambda_anti=spec['lam'],
            device=device)

        # Full dataset forward pass
        model.eval()
        X_all_t = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_all, hidden_all = model(X_all_t)

        final_cc = float(np.corrcoef(
            pred_all.cpu().numpy().ravel(), Y.ravel())[0, 1])

        h_flat = hidden_all.cpu().numpy().reshape(-1, hidden_dim)
        gamma_flat = probe_targets[:, gamma_idx]
        gamma_corr = float(np.max(np.abs(np.array([
            np.corrcoef(h_flat[:, d], gamma_flat)[0, 1]
            for d in range(hidden_dim)]))))

        probes = probe_mandatory(
            model, X, probe_targets, var_names, hidden_dim, device)

        # Bio-correlated dimension count
        dim_scores = np.zeros(hidden_dim)
        for j in range(probe_targets.shape[1]):
            for d in range(hidden_dim):
                r = abs(np.corrcoef(
                    h_flat[:, d], probe_targets[:, j])[0, 1])
                dim_scores[d] = max(dim_scores[d], r)
        n_bio = int(np.sum(dim_scores > 0.25))
        n_alien = hidden_dim - n_bio

        result = {
            'condition': cname,
            'description': spec['desc'],
            'final_cc': final_cc,
            'gamma_max_corr': gamma_corr,
            'n_bio_dims': n_bio,
            'n_alien_dims': n_alien,
            'mandatory_probes': probes,
            'epochs_trained': len(history['train_loss']),
            'final_train_loss': history['train_loss'][-1],
        }
        all_results[cname] = result

        logger.info("  CC=%.3f  gamma=%.3f  bio=%d/%d",
                     final_cc, gamma_corr, n_bio, hidden_dim)
        for vn, pr in probes.items():
            logger.info("    %s: dR2=%.3f max|r|=%.3f dims>0.3=%d",
                        vn, pr['delta_r2'],
                        pr['max_abs_correlation'],
                        pr['n_dims_above_0.3'])

        torch.save(model.state_dict(),
                   output_dir / f'{cname}_model.pt')

    # --- Summary ---
    logger.info("=" * 80)
    logger.info("ZOMBIE TRANSFER SUMMARY")
    logger.info("=" * 80)
    logger.info("%-30s %6s %8s %9s %6s %15s",
                'Condition', 'CC', 'gamma_r', 'bio_dims',
                'alien', 'Verdict')
    logger.info("-" * 80)

    for cname, r in all_results.items():
        cc, gr = r['final_cc'], r['gamma_max_corr']
        bd, ad = r['n_bio_dims'], r['n_alien_dims']
        if cc < 0.3:
            v = 'OUTPUT_FAILED'
        elif gr < 0.15 and cc > 0.5:
            v = 'ZOMBIE_FOUND!'
        elif gr > 0.5 and cc > 0.5:
            v = 'BIO_PRESERVED'
        elif gr < 0.3 and cc > 0.3:
            v = 'PARTIAL_ZOMBIE'
        else:
            v = 'AMBIGUOUS'
        logger.info("%-30s %6.3f %8.3f %9d %6d %15s",
                     cname, cc, gr, bd, ad, v)

    # --- Overall determination ---
    logger.info("=" * 80)
    logger.info("OVERALL DETERMINATION")
    logger.info("=" * 80)

    any_zombie = any(
        r['final_cc'] > 0.5 and r['gamma_max_corr'] < 0.15
        for r in all_results.values())

    baseline = all_results['C5_random_baseline']
    frozen = all_results['C1_frozen_transfer']
    warm = all_results['C3_warm_start']

    if any_zombie:
        zc = [k for k, r in all_results.items()
              if r['final_cc'] > 0.5 and r['gamma_max_corr'] < 0.15]
        logger.info("ZOMBIE PATHWAY FOUND via %s", zc)
        logger.info("  gamma_amp is NOT universally mandatory.")
        logger.info("  L5PC alien computation CAN serve hippocampus.")
    elif frozen['final_cc'] < 0.3:
        logger.info("L5PC ALIEN COMPUTATION IS INCOMPATIBLE")
        logger.info("  Frozen CC=%.3f -- alien dims cannot serve "
                     "hippocampal computation.", frozen['final_cc'])
        if warm['final_cc'] > 0.5 and warm['gamma_max_corr'] > 0.3:
            logger.info("  BUT warm-start (CC=%.3f) recovered biology "
                         "(gamma=%.3f).", warm['final_cc'],
                         warm['gamma_max_corr'])
            logger.info("  gamma_amp is MANDATORY: even from alien "
                         "basin, optimizer converges to biology.")
    else:
        logger.info("AMBIGUOUS -- additional experiments needed.")
        logger.info("  Frozen CC=%.3f, Warm CC=%.3f",
                     frozen['final_cc'], warm['final_cc'])

    out_path = output_dir / 'zombie_transfer_results.json'
    with open(out_path, 'w') as f:
        json.dump(_convert_numpy(all_results), f, indent=2)
    logger.info("Saved: %s", out_path)

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='L5PC zombie transfer test')
    parser.add_argument('--hippo-data-dir', required=True,
                        help='Hippocampal checkpoints_rates dir')
    parser.add_argument('--l5pc-checkpoint', required=True,
                        help='L5PC LSTM checkpoint (.pt)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    T0 = time.time()
    run_experiment(
        args.hippo_data_dir, args.l5pc_checkpoint,
        args.output_dir, args.hidden_dim, args.device)
    logger.info("Total time: %.1fs", time.time() - T0)
