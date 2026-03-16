"""Smoke test: Phase 1 modules import and run on synthetic data.

Run directly if pytest has torch DLL issues on Windows:
    python tests/test_phase1_smoke.py
"""
import numpy as np


def test_registry_core_probes_available():
    from l5pc.probing.registry import AVAILABLE_PROBES
    for core in ['ridge', 'mlp', 'sae', 'hardening', 'resample_ablation']:
        assert AVAILABLE_PROBES[core] is True, f"{core} should be available"


def test_mlp_probe_synthetic():
    from l5pc.probing.mlp_probe import mlp_delta_r2
    rng = np.random.RandomState(42)
    N, D = 100, 16
    hidden_trained = rng.randn(N, D).astype(np.float32)
    hidden_untrained = rng.randn(N, D).astype(np.float32)
    # Target correlates with first dim of trained
    target = hidden_trained[:, 0] + 0.1 * rng.randn(N)
    targets = target.reshape(-1, 1).astype(np.float32)
    results = mlp_delta_r2(hidden_trained, hidden_untrained, targets,
                           ['test_var'], epochs=5, n_splits=3)
    assert 'test_var' in results
    assert 'encoding_type' in results['test_var']


def test_sae_synthetic():
    from l5pc.probing.sae_probe import SparseAutoencoder, train_sae
    rng = np.random.RandomState(42)
    hidden = [rng.randn(50, 16).astype(np.float32)]
    sae, loss_history = train_sae(hidden, 16, expansion_factor=2,
                                   k=5, epochs=3, batch_size=32)
    assert len(loss_history) == 3
    assert loss_history[-1] < loss_history[0]  # Loss decreases


def test_hardening_imports():
    from l5pc.probing.hardening import hardened_probe
    from l5pc.probing.hardening.permutation import block_permute, adaptive_block_size
    from l5pc.probing.hardening.diagnostics import durbin_watson, effective_dof
    from l5pc.probing.hardening.corrections import fdr_correction, tost_zombie_test
    from l5pc.probing.hardening.frequency import frequency_resolved_r2
    from l5pc.probing.hardening.gap_cv import gap_temporal_cv
    # All imports succeed
    assert callable(hardened_probe)


def test_block_permute_preserves_shape():
    from l5pc.probing.hardening.permutation import block_permute
    rng = np.random.default_rng(42)
    y = rng.standard_normal(200)
    y_perm = block_permute(y, block_size=50, rng=rng)
    assert y_perm.shape == y.shape


def test_config_v3_constants():
    from l5pc.config import (
        SAE_K, SAE_LR, SAE_EPOCHS, SAE_BATCH_SIZE,
        MLP_PROBE_HIDDEN_DIM, MLP_PROBE_EPOCHS, MLP_PROBE_LR,
        HARDENING_N_BLOCK_PERMS, HARDENING_FDR_ALPHA,
        FREQUENCY_BANDS, CCA_N_COMPONENTS, TEMPORAL_WINDOW_SIZES,
    )
    assert SAE_K == 20
    assert MLP_PROBE_HIDDEN_DIM == 64
    assert HARDENING_N_BLOCK_PERMS == 500
    assert 'ultra_slow' in FREQUENCY_BANDS


if __name__ == '__main__':
    tests = [v for k, v in globals().items() if k.startswith('test_')]
    for t in tests:
        t()
        print(f'PASS: {t.__name__}')
    print(f'\nALL {len(tests)} TESTS PASSED')
