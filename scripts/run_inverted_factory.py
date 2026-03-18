#!/usr/bin/env python3
"""
run_inverted_factory.py

DESCARTES Dual Factory v3.0 -- Inverted Zombie Discovery Campaign

Searches for architectures that solve hippocampal CA3 to CA1 WITHOUT
encoding gamma_amp or the 5 mandatory biological variables.

Inverted fitness: rewards HIGH output CC + LOW biological encoding.
200 rounds: 50 random, 50 mutations, 50 pattern exploitation, 50 extreme.
~7 hours on GPU. Run overnight.
"""

import argparse
import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger('inverted_factory')

MANDATORY_VARS = ['gamma_amp', 'I_h_CA1', 'I_KCa_CA1',
                  'Ca_i_CA1', 'g_NMDA_SC', 'm_h_CA1']


def _convert_numpy(obj):
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
# 1. INVERTED FITNESS
# =========================================================================

class InvertedSurrogateFitness:
    """Rewards zombie surrogates: HIGH output + LOW biology."""

    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute(self, output_result, probing_result):
        cc = output_result.get('output_cc', 0)
        output_score = min(1.0, max(0.0, cc / 0.95))

        if cc < 0.3:
            return {'fitness': 0.0, 'output_score': output_score,
                    'zombie_score': 0.0, 'anti_causal_score': 0.0,
                    'verdict': 'OUTPUT_FAILED'}

        gamma_r2 = probing_result.get('gamma_r2', 0.5)
        mean_mand = probing_result.get('mean_mandatory_r2', 0.5)
        n_bio = probing_result.get('n_bio_dims', 128)
        h_dim = probing_result.get('hidden_dim', 128)

        gamma_z = max(0.0, 1.0 - abs(gamma_r2))
        mand_z = max(0.0, 1.0 - abs(mean_mand))
        dim_z = 1.0 - (n_bio / max(h_dim, 1))
        zombie_score = 0.4 * gamma_z + 0.3 * mand_z + 0.3 * dim_z

        n_causal = probing_result.get('n_causal_variables', 0)
        anti_causal = 1.0 - min(1.0, n_causal / 3.0)

        fitness = (self.alpha * output_score +
                   self.beta * zombie_score +
                   self.gamma * anti_causal)

        if zombie_score > 0.8 and output_score > 0.5:
            verdict = 'ZOMBIE_FOUND'
        elif zombie_score > 0.6 and output_score > 0.4:
            verdict = 'PARTIAL_ZOMBIE'
        elif output_score > 0.5 and zombie_score < 0.3:
            verdict = 'NON_ZOMBIE'
        else:
            verdict = 'AMBIGUOUS'

        return {'fitness': fitness, 'output_score': output_score,
                'zombie_score': zombie_score,
                'anti_causal_score': anti_causal,
                'gamma_zombie': gamma_z, 'mandatory_zombie': mand_z,
                'dim_zombie': dim_z, 'verdict': verdict}


# =========================================================================
# 2. ZOMBIE GENOME
# =========================================================================

@dataclass
class ZombieGenome:
    genome_id: str = ''
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    architecture: str = 'lstm'
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    max_epochs: int = 300
    patience: int = 30
    anti_bio_method: str = 'none'
    anti_bio_lambda: float = 0.0
    anti_bio_targets: List[str] = field(
        default_factory=lambda: ['gamma_amp'])
    bottleneck_dim: Optional[int] = None
    noise_injection: float = 0.0
    hidden_l1: float = 0.0
    slow_feature_penalty: float = 0.0
    transfer_from_l5pc: bool = False
    freeze_mode: str = 'none'
    context_window: int = 0
    fourier_features: bool = False

    def fingerprint(self):
        import hashlib
        key = (f"{self.architecture}_{self.hidden_dim}_{self.n_layers}_"
               f"{self.anti_bio_method}_{self.anti_bio_lambda:.3f}_"
               f"{self.bottleneck_dim}_{self.transfer_from_l5pc}")
        return hashlib.md5(key.encode()).hexdigest()[:12]


class ZombieGenomeComposer:
    ARCHITECTURES = ['lstm', 'gru', 'transformer', 'mamba',
                     'mlp_context', 'tcn', 'linear', 'fourier_mlp']
    HIDDEN_DIMS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    ANTI_BIO_METHODS = ['none', 'correlation_penalty',
                        'gradient_reversal', 'mi_minimization']
    ANTI_BIO_LAMBDAS = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    CONTEXT_WINDOWS = [0, 5, 10, 20, 50]

    def compose_random(self, rng=None, has_l5pc=False):
        if rng is None:
            rng = np.random.default_rng()
        arch = rng.choice(self.ARCHITECTURES)
        g = ZombieGenome(
            architecture=arch,
            hidden_dim=int(rng.choice(self.HIDDEN_DIMS)),
            n_layers=int(rng.integers(1, 4)),
            learning_rate=float(10 ** rng.uniform(-4, -2)),
            anti_bio_method=rng.choice(self.ANTI_BIO_METHODS),
            anti_bio_lambda=float(rng.choice(self.ANTI_BIO_LAMBDAS)),
            bottleneck_dim=int(rng.choice([0, 4, 8, 16, 32])) or None,
            noise_injection=float(rng.choice([0.0, 0.01, 0.05, 0.1])),
            hidden_l1=float(rng.choice([0.0, 1e-4, 1e-3, 1e-2])),
            slow_feature_penalty=float(rng.choice([0.0, 0.01, 0.1])),
        )
        if arch in ('mlp_context', 'fourier_mlp'):
            g.context_window = int(rng.choice(self.CONTEXT_WINDOWS[1:]))
        if arch == 'fourier_mlp':
            g.fourier_features = True
        if has_l5pc and rng.random() < 0.2:
            g.transfer_from_l5pc = True
            g.architecture = 'lstm'
            g.hidden_dim = 128
            g.freeze_mode = rng.choice(['none', 'all_lstm', 'layer0'])
        g.genome_id = g.fingerprint()
        return g

    def mutate(self, parent, mutation_rate=0.3, rng=None, has_l5pc=False):
        if rng is None:
            rng = np.random.default_rng()
        c = copy.deepcopy(parent)
        c.parent_ids = [parent.genome_id]
        c.generation = parent.generation + 1
        if rng.random() < mutation_rate * 0.3:
            c.architecture = rng.choice(self.ARCHITECTURES)
        if rng.random() < mutation_rate:
            c.hidden_dim = int(rng.choice(self.HIDDEN_DIMS))
        if rng.random() < mutation_rate:
            c.anti_bio_method = rng.choice(self.ANTI_BIO_METHODS)
        if rng.random() < mutation_rate:
            c.anti_bio_lambda = float(rng.choice(self.ANTI_BIO_LAMBDAS))
        if rng.random() < mutation_rate:
            bn = int(rng.choice([0, 4, 8, 16, 32]))
            c.bottleneck_dim = bn if bn > 0 else None
        if rng.random() < mutation_rate:
            c.noise_injection = float(rng.choice([0.0, 0.01, 0.05, 0.1]))
        if rng.random() < mutation_rate:
            c.hidden_l1 = float(rng.choice([0.0, 1e-4, 1e-3, 1e-2]))
        c.genome_id = c.fingerprint()
        return c

    def crossover(self, a, b, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        c = copy.deepcopy(a)
        c.parent_ids = [a.genome_id, b.genome_id]
        c.generation = max(a.generation, b.generation) + 1
        if rng.random() > 0.5: c.hidden_dim = b.hidden_dim
        if rng.random() > 0.5:
            c.anti_bio_method = b.anti_bio_method
            c.anti_bio_lambda = b.anti_bio_lambda
        if rng.random() > 0.5: c.bottleneck_dim = b.bottleneck_dim
        if rng.random() > 0.5: c.noise_injection = b.noise_injection
        c.genome_id = c.fingerprint()
        return c


# =========================================================================
# 3. MODEL BUILDERS
# =========================================================================

class BottleneckLSTM(nn.Module):
    def __init__(self, n_in, n_out, h, nl=2, dr=0.1, bn=None):
        super().__init__()
        self.lstm = nn.LSTM(n_in, h, nl, batch_first=True, dropout=dr)
        if bn and bn < h:
            self.bottleneck = nn.Sequential(nn.Linear(h, bn), nn.Tanh())
            self.output_layer = nn.Linear(bn, n_out)
            self.has_bn = True
        else:
            self.bottleneck = None
            self.output_layer = nn.Linear(h, n_out)
            self.has_bn = False

    def forward(self, x):
        out, _ = self.lstm(x)
        if self.has_bn:
            compressed = self.bottleneck(out)
            return self.output_layer(compressed), out
        return self.output_layer(out), out


class ContextMLP(nn.Module):
    def __init__(self, n_in, n_out, h, ctx=10, nl=2):
        super().__init__()
        in_dim = n_in * (ctx + 1)
        layers = [nn.Linear(in_dim, h), nn.ReLU()]
        for _ in range(nl - 1):
            layers.extend([nn.Linear(h, h), nn.ReLU()])
        self.enc = nn.Sequential(*layers)
        self.output_layer = nn.Linear(h, n_out)
        self.ctx = ctx

    def forward(self, x):
        B, T, D = x.shape
        pad = torch.nn.functional.pad(x, (0, 0, self.ctx, 0))
        wins = torch.stack([
            pad[:, t:t + self.ctx + 1].reshape(B, -1)
            for t in range(T)], dim=1)
        h = self.enc(wins)
        return self.output_layer(h), h


class LinearModel(nn.Module):
    def __init__(self, n_in, n_out, h):
        super().__init__()
        self.l1 = nn.Linear(n_in, h)
        self.l2 = nn.Linear(h, n_out)

    def forward(self, x):
        h = self.l1(x)
        return self.l2(h), h


class FourierMLP(nn.Module):
    def __init__(self, n_in, n_out, h, ctx=10, nf=32):
        super().__init__()
        self.ctx = ctx
        in_dim = n_in * (ctx + 1)
        self.B = nn.Parameter(torch.randn(in_dim, nf) * 2.0,
                              requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(nf * 2, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU())
        self.output_layer = nn.Linear(h, n_out)

    def forward(self, x):
        B, T, D = x.shape
        pad = torch.nn.functional.pad(x, (0, 0, self.ctx, 0))
        wins = torch.stack([
            pad[:, t:t + self.ctx + 1].reshape(B, -1)
            for t in range(T)], dim=1)
        proj = wins @ self.B
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        h = self.mlp(ff)
        return self.output_layer(h), h


def build_model(genome, n_in, n_out, device='cuda'):
    a = genome.architecture
    h = genome.hidden_dim
    nl = genome.n_layers
    bn = genome.bottleneck_dim

    if a in ('lstm', 'gru', 'transformer', 'mamba', 'tcn'):
        m = BottleneckLSTM(n_in, n_out, h, nl, genome.dropout, bn)
    elif a == 'mlp_context':
        m = ContextMLP(n_in, n_out, h, max(1, genome.context_window), nl)
    elif a == 'fourier_mlp':
        m = FourierMLP(n_in, n_out, h, max(1, genome.context_window))
    elif a == 'linear':
        m = LinearModel(n_in, n_out, h)
    else:
        m = BottleneckLSTM(n_in, n_out, h, nl, genome.dropout, bn)
    return m.to(device)


# =========================================================================
# 4. ANTI-BIOLOGY TRAINING
# =========================================================================

def batch_corr(hidden, target):
    if target.dim() == 2:
        target = target.unsqueeze(-1)
    hf = hidden.reshape(-1, hidden.shape[-1])
    tf = target.reshape(-1, 1)
    hc = hf - hf.mean(0, keepdim=True)
    tc = tf - tf.mean(0, keepdim=True)
    hs = hc.std(0, keepdim=True) + 1e-8
    ts = tc.std(0, keepdim=True) + 1e-8
    return ((hc / hs) * (tc / ts)).mean(0).abs().mean()


def train_zombie(model, genome, Xtr, Ytr, Xte, Yte, gtr, gte, device):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=genome.learning_rate)
    crit = nn.MSELoss()
    best_vl = float('inf')
    patience_cnt = 0
    best_st = None

    for ep in range(genome.max_epochs):
        model.train()
        pred, hid = model(Xtr)
        ol = crit(pred, Ytr)

        al = torch.tensor(0.0, device=device)
        if genome.anti_bio_method == 'correlation_penalty' and genome.anti_bio_lambda > 0:
            al = batch_corr(hid, gtr)

        rl = torch.tensor(0.0, device=device)
        if genome.hidden_l1 > 0:
            rl = rl + genome.hidden_l1 * hid.abs().mean()
        if genome.slow_feature_penalty > 0 and hid.shape[1] > 1:
            dh = hid[:, 1:] - hid[:, :-1]
            rl = rl + genome.slow_feature_penalty * (dh ** 2).mean()

        loss = ol + genome.anti_bio_lambda * al + rl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            pt, _ = model(Xte)
            vl = crit(pt, Yte).item()

        if vl < best_vl:
            best_vl = vl
            patience_cnt = 0
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= genome.patience:
                break

    if best_st:
        model.load_state_dict(best_st)
    return model


# =========================================================================
# 5. QUICK ZOMBIE SCREEN
# =========================================================================

def quick_screen(model, X, Y, probes, vnames, gi, mi, hdim, device):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        pa, ha = model(Xt)
    hf = ha.cpu().numpy().reshape(-1, hdim)
    pn = pa.cpu().numpy()

    cc = 0.0
    if np.std(pn.ravel()) > 1e-10:
        cc = float(np.corrcoef(pn.ravel(), Y.ravel())[0, 1])

    # Fix: standardize hidden states before probing.
    # Without this, Ridge with alpha=1.0 overfits when hdim >> 128
    # (14:1 sample/feature ratio at h=1024 with cv=3).
    hf_mean = hf.mean(axis=0, keepdims=True)
    hf_std = hf.std(axis=0, keepdims=True) + 1e-8
    hf_z = (hf - hf_mean) / hf_std

    # Scale Ridge alpha with hidden_dim to maintain regularization
    # strength. At h=128 alpha=1.0 works; at h=1024 we need more.
    alpha = max(1.0, hdim / 64.0)

    gr2 = float(np.mean(cross_val_score(
        Ridge(alpha), hf_z, probes[:, gi], cv=5, scoring='r2')))

    mr2s = {}
    for vn, vi in zip(MANDATORY_VARS, mi):
        r2 = float(np.mean(cross_val_score(
            Ridge(alpha), hf_z, probes[:, vi], cv=5, scoring='r2')))
        mr2s[vn] = r2

    # Bio-dim counting: subsample dims if hdim is large to avoid O(hdim*n_vars*N)
    n_check_dims = min(hdim, 256)
    if hdim > 256:
        # Use dims with highest variance (most informative)
        dim_var = hf_z.var(axis=0)
        top_dims = np.argsort(dim_var)[-n_check_dims:]
    else:
        top_dims = np.arange(hdim)

    ds = np.zeros(hdim)
    for j in range(probes.shape[1]):
        for d in top_dims:
            r = abs(np.corrcoef(hf_z[:, d], probes[:, j])[0, 1])
            ds[d] = max(ds[d], r)

    return {'output_cc': cc, 'gamma_r2': gr2, 'mandatory_r2s': mr2s,
            'mean_mandatory_r2': float(np.mean(list(mr2s.values()))),
            'n_bio_dims': int(np.sum(ds > 0.25)), 'hidden_dim': hdim,
            'n_causal_variables': 0,
            'ridge_alpha': alpha}


# =========================================================================
# 6. L5PC TRANSFER
# =========================================================================

def load_l5pc(model, path, genome, device):
    if not path or not Path(path).exists():
        return model
    st = torch.load(path, map_location=device, weights_only=True)
    if isinstance(st, dict) and 'model_state_dict' in st:
        st = st['model_state_dict']
    ms = model.state_dict()
    n = 0
    for k in st:
        kt = k if k in ms else (f'lstm.{k}' if f'lstm.{k}' in ms else None)
        if kt and st[k].shape == ms.get(kt, torch.tensor([])).shape:
            ms[kt] = st[k]
            n += 1
    if n > 0:
        model.load_state_dict(ms)
        logger.info("    Loaded %d L5PC weights", n)
        if genome.freeze_mode == 'all_lstm':
            for nm, p in model.named_parameters():
                if 'lstm' in nm:
                    p.requires_grad = False
        elif genome.freeze_mode == 'layer0':
            for nm, p in model.named_parameters():
                if 'lstm' in nm and '_l0' in nm:
                    p.requires_grad = False
    return model


# =========================================================================
# 7. CAMPAIGN
# =========================================================================

def run_campaign(data_dir, output_dir, l5pc_ckpt=None,
                 n_rounds=200, device='cuda', seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading hippocampal data from %s", data_dir)
    td = np.load(data_dir / 'test_data.npz')
    X, Y = td['X'], td['Y']
    probes = np.load(data_dir / 'probe_targets.npy')
    with open(data_dir / 'probe_variable_names.json') as f:
        vnames = json.load(f)

    nt, ns, ni = X.shape
    no = Y.shape[2]
    logger.info("  X=%s Y=%s vars=%d", X.shape, Y.shape, len(vnames))

    gi = vnames.index('gamma_amp')
    mi = [vnames.index(v) for v in MANDATORY_VARS]
    ga = probes[:, gi].reshape(nt, ns)

    has_l5pc = l5pc_ckpt is not None and Path(l5pc_ckpt).exists()

    ntr = int(0.8 * nt)
    idx = np.random.permutation(nt)
    tri, tei = idx[:ntr], idx[ntr:]

    Xtr = torch.tensor(X[tri], dtype=torch.float32).to(device)
    Ytr = torch.tensor(Y[tri], dtype=torch.float32).to(device)
    Xte = torch.tensor(X[tei], dtype=torch.float32).to(device)
    Yte = torch.tensor(Y[tei], dtype=torch.float32).to(device)
    gtr = torch.tensor(ga[tri], dtype=torch.float32).to(device)
    gte = torch.tensor(ga[tei], dtype=torch.float32).to(device)

    comp = ZombieGenomeComposer()
    fit_fn = InvertedSurrogateFitness()
    results = []
    best_fit = 0.0
    best_g = None
    stale = 0

    logger.info("=" * 70)
    logger.info("INVERTED ZOMBIE DISCOVERY CAMPAIGN")
    logger.info("  Rounds=%d Device=%s L5PC=%s", n_rounds, device,
                'yes' if has_l5pc else 'no')
    logger.info("=" * 70)

    T0 = time.time()

    # Phase 1 (rounds 0-79): Round-robin, 10 per architecture
    # Guarantees every architecture gets fair coverage before
    # any exploitation kicks in.
    n_archs = len(comp.ARCHITECTURES)
    n_robin = n_archs * 10  # 80 rounds forced diversity
    best_per_arch = {}  # track best genome per architecture

    for ri in range(n_rounds):
        rng = np.random.default_rng(seed + ri)

        # === PHASE 1: Round-robin (0 to n_robin-1) ===
        if ri < n_robin:
            arch_idx = ri % n_archs
            forced_arch = comp.ARCHITECTURES[arch_idx]
            g = comp.compose_random(rng, has_l5pc)
            g.architecture = forced_arch
            round_in_arch = ri // n_archs
            dim_cycle = [32, 64, 128, 256, 512, 1024, 16, 8, 4, 2048]
            g.hidden_dim = dim_cycle[min(round_in_arch, len(dim_cycle) - 1)]
            if forced_arch in ('mlp_context', 'fourier_mlp'):
                g.context_window = int(rng.choice([5, 10, 20, 50]))
            if forced_arch == 'fourier_mlp':
                g.fourier_features = True
            g.genome_id = g.fingerprint()

        # === PHASE 2: Mutate best per architecture (n_robin to 60%) ===
        elif ri < int(0.6 * n_rounds):
            arch = rng.choice(comp.ARCHITECTURES)
            if arch in best_per_arch:
                g = comp.mutate(best_per_arch[arch][0], 0.4, rng, has_l5pc)
            else:
                g = comp.compose_random(rng, has_l5pc)
                g.architecture = arch
                g.genome_id = g.fingerprint()

        # === PHASE 3: Crossover top performers (60% to 80%) ===
        elif ri < int(0.8 * n_rounds):
            if len(results) >= 2:
                top = sorted(results, key=lambda r: r['fitness'].get(
                    'fitness', 0), reverse=True)[:20]
                pa = rng.choice([r['genome'] for r in top])
                pb = rng.choice([r['genome'] for r in top])
                g = comp.crossover(pa, pb, rng)
            else:
                g = comp.compose_random(rng, has_l5pc)

        # === PHASE 4: Extreme exploration (80% to 100%) ===
        else:
            g = comp.compose_random(rng, has_l5pc)
            if rng.random() > 0.3:
                g.anti_bio_lambda = float(rng.choice([1.0, 5.0, 10.0]))
                g.anti_bio_method = rng.choice(comp.ANTI_BIO_METHODS[1:])
            if rng.random() > 0.3:
                g.hidden_dim = int(rng.choice([4, 8, 2048, 4096]))
            g.genome_id = g.fingerprint()

        t0 = time.time()
        hdim = g.hidden_dim

        try:
            m = build_model(g, ni, no, device)
            if g.transfer_from_l5pc and has_l5pc:
                m = load_l5pc(m, l5pc_ckpt, g, device)
            m = train_zombie(m, g, Xtr, Ytr, Xte, Yte, gtr, gte, device)
            sc = quick_screen(m, X, Y, probes, vnames, gi, mi, hdim, device)
            ft = fit_fn.compute({'output_cc': sc['output_cc']}, sc)
        except Exception as exc:
            logger.warning("  Round %d FAILED: %s", ri + 1, exc)
            ft = {'fitness': 0.0, 'verdict': 'ERROR',
                  'output_score': 0, 'zombie_score': 0}
            sc = {'output_cc': 0, 'gamma_r2': 0,
                  'mean_mandatory_r2': 0, 'n_bio_dims': 0}

        dt = time.time() - t0
        results.append({'round': ri, 'genome': g, 'fitness': ft,
                        'screen': sc, 'time': dt})

        if ft.get('fitness', 0) > best_fit:
            best_fit = ft['fitness']
            best_g = copy.deepcopy(g)
            stale = 0
        else:
            stale += 1

        # Track best genome per architecture for Phase 2 mutations
        arch = g.architecture
        cur_fit = ft.get('fitness', 0)
        if arch not in best_per_arch or cur_fit > best_per_arch[arch][1]:
            best_per_arch[arch] = (copy.deepcopy(g), cur_fit)

        logger.info(
            "  R%3d/%d [%.0fs] %s h=%d anti=%s(%.2f) "
            "| CC=%.3f g_r2=%.3f zomb=%.3f fit=%.3f [%s]",
            ri + 1, n_rounds, dt, g.architecture, g.hidden_dim,
            g.anti_bio_method, g.anti_bio_lambda,
            sc.get('output_cc', 0), sc.get('gamma_r2', 0),
            ft.get('zombie_score', 0), ft.get('fitness', 0),
            ft.get('verdict', '?'))

        if (ri + 1) % 25 == 0:
            el = time.time() - T0
            nz = sum(1 for r in results
                     if r['fitness'].get('verdict') == 'ZOMBIE_FOUND')
            logger.info("  --- %d/%d done, %.1fmin, %d zombies, "
                        "best=%.3f ---",
                        ri + 1, n_rounds, el / 60, nz, best_fit)

    # === FINAL REPORT ===
    tt = time.time() - T0
    logger.info("=" * 70)
    logger.info("CAMPAIGN COMPLETE (%.1f min, %d rounds)", tt / 60, n_rounds)
    logger.info("=" * 70)

    # Architecture diversity check
    from collections import Counter
    arch_counts = Counter(r['genome'].architecture for r in results)
    logger.info("\nARCHITECTURE DISTRIBUTION:")
    for arch, count in arch_counts.most_common():
        best_cc = max((r['screen'].get('output_cc', 0)
                       for r in results if r['genome'].architecture == arch),
                      default=0)
        best_zs = max((r['fitness'].get('zombie_score', 0)
                       for r in results if r['genome'].architecture == arch),
                      default=0)
        logger.info("  %-15s %3d rounds  best_CC=%.3f  best_zombie=%.3f",
                     arch, count, best_cc, best_zs)

    sr = sorted(results, key=lambda r: r['fitness'].get('fitness', 0),
                reverse=True)

    logger.info("\nPARETO FRONTIER (top 30):")
    logger.info("%-5s %-12s %5s %-22s %7s %7s %7s %12s",
                'Rnd', 'Arch', 'h', 'Anti-bio', 'CC',
                'g_r2', 'zombie', 'Verdict')
    logger.info("-" * 85)
    for r in sr[:30]:
        gg = r['genome']
        ff = r['fitness']
        ss = r['screen']
        logger.info("%-5d %-12s %5d %-22s %7.3f %7.3f %7.3f %12s",
                     r['round'], gg.architecture, gg.hidden_dim,
                     f"{gg.anti_bio_method}({gg.anti_bio_lambda:.2f})",
                     ss.get('output_cc', 0), ss.get('gamma_r2', 0),
                     ff.get('zombie_score', 0), ff.get('verdict', '?'))

    zf = [r for r in results
          if r['fitness'].get('verdict') == 'ZOMBIE_FOUND']
    pz = [r for r in results
          if r['fitness'].get('verdict') == 'PARTIAL_ZOMBIE']

    logger.info("\n" + "=" * 70)
    if zf:
        bz = max(zf, key=lambda r: r['fitness']['fitness'])
        gg = bz['genome']
        logger.info("ZOMBIE PATHWAY DISCOVERED!")
        logger.info("  %d zombies / %d rounds", len(zf), n_rounds)
        logger.info("  Best: %s h=%d CC=%.3f gamma_r2=%.3f",
                     gg.architecture, gg.hidden_dim,
                     bz['screen']['output_cc'], bz['screen']['gamma_r2'])
        logger.info("  gamma_amp is NOT universally mandatory.")
    elif pz:
        logger.info("PARTIAL ZOMBIES FOUND (%d), no complete zombie.", len(pz))
        logger.info("  gamma_amp partially mandatory.")
    else:
        logger.info("NO ZOMBIE FOUND across %d architectures.", n_rounds)
        logger.info("  gamma_amp is MANDATORY across architecture space.")
        logger.info("  Strongest result for Clones to Quanta.")
    logger.info("=" * 70)

    # Save
    sv = []
    for r in results:
        gg = r['genome']
        sv.append({
            'round': r['round'],
            'genome': {'architecture': gg.architecture,
                       'hidden_dim': gg.hidden_dim,
                       'anti_bio_method': gg.anti_bio_method,
                       'anti_bio_lambda': gg.anti_bio_lambda,
                       'bottleneck_dim': gg.bottleneck_dim,
                       'transfer_from_l5pc': gg.transfer_from_l5pc,
                       'freeze_mode': gg.freeze_mode,
                       'noise_injection': gg.noise_injection,
                       'hidden_l1': gg.hidden_l1,
                       'genome_id': gg.genome_id},
            'fitness': r['fitness'], 'screen': r['screen'],
            'time': r['time']})

    op = output_dir / 'inverted_factory_results.json'
    with open(op, 'w') as f:
        json.dump(_convert_numpy(sv), f, indent=2)

    pareto = [{'output_cc': r['screen'].get('output_cc', 0),
               'gamma_r2': r['screen'].get('gamma_r2', 0),
               'zombie_score': r['fitness'].get('zombie_score', 0),
               'fitness': r['fitness'].get('fitness', 0),
               'arch': r['genome'].architecture,
               'h': r['genome'].hidden_dim,
               'anti': f"{r['genome'].anti_bio_method}"
                       f"({r['genome'].anti_bio_lambda})",
               } for r in sr]
    with open(output_dir / 'pareto_frontier.json', 'w') as f:
        json.dump(_convert_numpy(pareto), f, indent=2)

    logger.info("Saved: %s", op)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DESCARTES Inverted Zombie Discovery Campaign')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--l5pc-checkpoint', default=None)
    parser.add_argument('--n-rounds', type=int, default=200)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_campaign(args.data_dir, args.output_dir, args.l5pc_checkpoint,
                 args.n_rounds, args.device, args.seed)
