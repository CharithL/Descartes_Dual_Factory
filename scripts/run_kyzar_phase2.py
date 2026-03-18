#!/usr/bin/env python3
"""
run_phase2.py

DESCARTES Circuit 6 Phase 2: Train LSTM surrogates per session.
Gap cross-validation (70/10/20 chronological split).
Hidden sizes: [32, 64, 128]. Output gate: CC >= 0.3.
Also extracts hidden states and LSTM gate activations.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('phase2')


class LSTMSurrogate(nn.Module):
    def __init__(self, n_in, n_out, hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        dr = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(n_in, hidden_dim, n_layers,
                            batch_first=True, dropout=dr)
        self.output_layer = nn.Linear(hidden_dim, n_out)
        self.hidden_dim = hidden_dim

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                         enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)
        pred = self.output_layer(out)
        return pred, out


def load_session_data(session_dir):
    session_dir = Path(session_dir)
    X_data = dict(np.load(session_dir / 'X_trials.npz'))
    Y_data = dict(np.load(session_dir / 'Y_trials.npz'))
    with open(session_dir / 'metadata.json') as f:
        meta = json.load(f)
    return X_data, Y_data, meta


def gap_cv_split(n_trials, train_frac=0.7, gap_frac=0.1):
    n_train = int(n_trials * train_frac)
    n_gap = int(n_trials * gap_frac)
    train_idx = list(range(n_train))
    test_idx = list(range(n_train + n_gap, n_trials))
    return train_idx, test_idx


def collate_trials(X_data, Y_data, indices, device='cpu'):
    X_list = [torch.tensor(X_data[f'trial_{i}'], dtype=torch.float32)
              for i in indices]
    Y_list = [torch.tensor(Y_data[f'trial_{i}'], dtype=torch.float32)
              for i in indices]
    lengths = torch.tensor([x.shape[0] for x in X_list])
    X_pad = pad_sequence(X_list, batch_first=True).to(device)
    Y_pad = pad_sequence(Y_list, batch_first=True).to(device)
    return X_pad, Y_pad, lengths


def train_model(model, X_data, Y_data, train_idx, test_idx,
                max_epochs=300, patience=20, lr=1e-3, device='cpu'):
    X_tr, Y_tr, L_tr = collate_trials(X_data, Y_data, train_idx, device)
    X_te, Y_te, L_te = collate_trials(X_data, Y_data, test_idx, device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_vl = float('inf')
    patience_cnt = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': []}

    for ep in range(max_epochs):
        model.train()
        pred, _ = model(X_tr, L_tr)
        mask = torch.zeros_like(pred, dtype=torch.bool)
        for i, l in enumerate(L_tr):
            mask[i, :l] = True
        loss = crit(pred[mask], Y_tr[mask])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            pred_te, _ = model(X_te, L_te)
            mask_te = torch.zeros_like(pred_te, dtype=torch.bool)
            for i, l in enumerate(L_te):
                mask_te[i, :l] = True
            vl = crit(pred_te[mask_te], Y_te[mask_te]).item()

        history['train_loss'].append(float(loss.item()))
        history['val_loss'].append(float(vl))

        if vl < best_vl:
            best_vl = vl
            patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_te, _ = model(X_te, L_te)
    all_pred, all_true = [], []
    for i, l in enumerate(L_te):
        all_pred.append(pred_te[i, :l].cpu().numpy())
        all_true.append(Y_te[i, :l].cpu().numpy())
    pred_flat = np.concatenate(all_pred).ravel()
    true_flat = np.concatenate(all_true).ravel()

    cc = 0.0
    if np.std(pred_flat) > 1e-10 and np.std(true_flat) > 1e-10:
        cc = float(np.corrcoef(pred_flat, true_flat)[0, 1])

    return model, cc, len(history['train_loss']), history


def extract_hidden_states(model, X_data, n_trials, device='cpu'):
    model.eval()
    hidden_dict = {}
    for ti in range(n_trials):
        x = torch.tensor(X_data['trial_' + str(ti)],
                         dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, h = model(x)
        hidden_dict['trial_' + str(ti)] = h[0].cpu().numpy()
    return hidden_dict


def extract_gate_activations(model, X_data, n_trials, device='cpu'):
    model.eval()
    hdim = model.hidden_dim

    W_ih = model.lstm.weight_ih_l0.data
    W_hh = model.lstm.weight_hh_l0.data
    b_ih = model.lstm.bias_ih_l0.data
    b_hh = model.lstm.bias_hh_l0.data

    W_ii, W_if, W_ig, W_io = W_ih.chunk(4, 0)
    W_hi, W_hf, W_hg, W_ho = W_hh.chunk(4, 0)
    b_ii, b_if, b_ig, b_io = b_ih.chunk(4, 0)
    b_hi, b_hf, b_hg, b_ho = b_hh.chunk(4, 0)

    gates_dict = {}
    for ti in range(n_trials):
        x = torch.tensor(X_data['trial_' + str(ti)],
                         dtype=torch.float32).unsqueeze(0).to(device)
        T = x.shape[1]

        forget_g, input_g, output_g, cell_s = [], [], [], []
        h_t = torch.zeros(1, hdim, device=device)
        c_t = torch.zeros(1, hdim, device=device)

        for t in range(T):
            x_t = x[0, t:t+1]
            i_t = torch.sigmoid(x_t @ W_ii.T + b_ii + h_t @ W_hi.T + b_hi)
            f_t = torch.sigmoid(x_t @ W_if.T + b_if + h_t @ W_hf.T + b_hf)
            g_t = torch.tanh(x_t @ W_ig.T + b_ig + h_t @ W_hg.T + b_hg)
            o_t = torch.sigmoid(x_t @ W_io.T + b_io + h_t @ W_ho.T + b_ho)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            forget_g.append(f_t.detach().cpu().numpy())
            input_g.append(i_t.detach().cpu().numpy())
            output_g.append(o_t.detach().cpu().numpy())
            cell_s.append(c_t.detach().cpu().numpy())

        prefix = 'trial_' + str(ti)
        gates_dict[prefix + '_forget'] = np.concatenate(forget_g, axis=0)
        gates_dict[prefix + '_input'] = np.concatenate(input_g, axis=0)
        gates_dict[prefix + '_output'] = np.concatenate(output_g, axis=0)
        gates_dict[prefix + '_cell'] = np.concatenate(cell_s, axis=0)

    return gates_dict


def process_session(session_dir, model_dir, hidden_dims=(32, 64, 128),
                    device='cpu'):
    session_dir = Path(session_dir)
    model_dir = Path(model_dir)

    X_data, Y_data, meta = load_session_data(session_dir)
    n_trials = meta['n_trials']
    n_in = X_data['trial_0'].shape[1]
    n_out = Y_data['trial_0'].shape[1]

    train_idx, test_idx = gap_cv_split(n_trials)
    log.info("  Split: %d train, %d test (gap=%d)",
             len(train_idx), len(test_idx),
             n_trials - len(train_idx) - len(test_idx))

    results = {}
    for hdim in hidden_dims:
        log.info("  Training h=%d...", hdim)
        out_dir = model_dir / ('lstm_h' + str(hdim))
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(42)
        model = LSTMSurrogate(n_in, n_out, hdim).to(device)
        model, cc, epochs, history = train_model(
            model, X_data, Y_data, train_idx, test_idx, device=device)

        passed = cc >= 0.3
        log.info("    CC=%.3f (%d epochs) %s",
                 cc, epochs, "PASS" if passed else "FAIL")

        torch.save(model.state_dict(), out_dir / 'trained_model.pt')

        torch.manual_seed(99)
        model_u = LSTMSurrogate(n_in, n_out, hdim).to(device)
        torch.save(model_u.state_dict(), out_dir / 'untrained_model.pt')

        with open(out_dir / 'training_history.json', 'w') as f:
            json.dump(history, f)

        val_info = {
            'output_cc': float(cc), 'passed_gate': passed,
            'epochs': epochs, 'hidden_dim': hdim,
            'n_in': n_in, 'n_out': n_out,
            'n_train': len(train_idx), 'n_test': len(test_idx),
        }
        with open(out_dir / 'output_validation.json', 'w') as f:
            json.dump(val_info, f, indent=2)

        if passed:
            log.info("    Extracting hidden states...")
            h_trained = extract_hidden_states(model, X_data, n_trials, device)
            np.savez(out_dir / 'hidden_trained.npz', **h_trained)

            h_untrained = extract_hidden_states(model_u, X_data, n_trials, device)
            np.savez(out_dir / 'hidden_untrained.npz', **h_untrained)

            log.info("    Extracting gate activations...")
            gates = extract_gate_activations(model, X_data, n_trials, device)
            np.savez(out_dir / 'gates_trained.npz', **gates)

        results[hdim] = val_info

    return results


def main():
    parser = argparse.ArgumentParser(
        description='DESCARTES Kyzar Phase 2: LSTM Training')
    parser.add_argument('--processed-dir', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--hidden-dims', nargs='+', type=int,
                        default=[32, 64, 128])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--subjects', nargs='*', default=None)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    model_base = Path(args.model_dir)

    if args.subjects:
        session_dirs = [processed_dir / ('session_sub' + s + '_ses2')
                        for s in args.subjects]
    else:
        session_dirs = sorted(processed_dir.glob('session_sub*_ses2'))

    log.info("Phase 2: Training on %d sessions, device=%s",
             len(session_dirs), args.device)
    t0 = time.time()

    all_results = {}
    for sd in session_dirs:
        sub_id = sd.name.split('_')[1].replace('sub', '')
        log.info("=" * 60)
        log.info("Subject %s", sub_id)

        md = model_base / sd.name
        try:
            results = process_session(
                sd, md, tuple(args.hidden_dims), args.device)
            all_results[sub_id] = results
        except Exception as exc:
            log.error("  FAILED: %s", exc)
            import traceback
            traceback.print_exc()

    dt = time.time() - t0

    log.info("\n" + "=" * 70)
    log.info("PHASE 2 COMPLETE (%.1f min)", dt / 60)
    log.info("=" * 70)

    header = "Subject "
    for h in args.hidden_dims:
        header += "  h=" + str(h).rjust(4)
    log.info(header)
    log.info("-" * 70)
    for sub_id, res in sorted(all_results.items(), key=lambda x: int(x[0])):
        line = "sub-" + sub_id.ljust(4)
        for h in args.hidden_dims:
            if h in res:
                cc = res[h]['output_cc']
                p = '*' if res[h]['passed_gate'] else ' '
                line += "  " + "{:.3f}".format(cc) + p
            else:
                line += "    N/A"
        log.info(line)


if __name__ == '__main__':
    main()
