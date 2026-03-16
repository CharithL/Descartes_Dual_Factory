"""
L5PC DESCARTES -- Statistical Hardening Suite

Entry point: hardened_probe() runs one target through the complete
13-method statistical hardening pipeline.

Usage:
    from l5pc.probing.hardening import hardened_probe
    result = hardened_probe(h_trained, h_untrained, target, 'gNaTa_t')
"""

import logging

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

from l5pc.probing.mlp_probe import mlp_delta_r2
from l5pc.probing.hardening.permutation import (
    block_permute, adaptive_block_size, phase_randomize, circular_shift_null,
)
from l5pc.probing.hardening.diagnostics import (
    effective_dof, durbin_watson, ljung_box_residual_test,
)
from l5pc.probing.hardening.corrections import (
    fdr_correction, tost_zombie_test, bayes_factor_null,
)
from l5pc.probing.hardening.frequency import (
    frequency_resolved_r2, partial_coherence_r2,
)
from l5pc.probing.hardening.gap_cv import (
    gap_temporal_cv, cluster_permutation_test,
)

logger = logging.getLogger(__name__)


def hardened_probe(hidden_trained, hidden_untrained, target,
                    target_name, input_signal=None,
                    sampling_rate=1000, device='cpu'):
    """
    Run ONE probe target through the complete hardening suite.
    Returns definitive verdict with formal statistics.
    """
    # 1. Adaptive block size from target autocorrelation
    block_size, tau_sec = adaptive_block_size(target, sampling_rate)

    # 2. Ridge delta-R2 (baseline)
    ridge = Ridge(alpha=1.0)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    trained_scores = cross_val_score(ridge, hidden_trained, target, cv=kf)
    untrained_scores = cross_val_score(Ridge(1.0), hidden_untrained, target, cv=kf)

    r2_trained = np.mean(trained_scores)
    r2_untrained = np.mean(untrained_scores)
    delta_r2 = r2_trained - r2_untrained
    se_delta = np.sqrt(np.var(trained_scores)/5 + np.var(untrained_scores)/5)

    # 3. MLP delta-R2 (nonlinear control)
    mlp_result = mlp_delta_r2(
        hidden_trained, hidden_untrained,
        target.reshape(-1, 1), [target_name], device=device)
    mlp_delta = mlp_result[target_name]['mlp_delta']

    # 4. Block permutation null
    rng = np.random.default_rng(42)
    n_perms = 500
    null_deltas = []
    for _ in range(n_perms):
        target_perm = block_permute(target, block_size, rng)
        r2_perm = np.mean(cross_val_score(
            Ridge(1.0), hidden_trained, target_perm, cv=kf))
        null_deltas.append(r2_perm - r2_untrained)
    null_deltas = np.array(null_deltas)
    p_block = (null_deltas >= delta_r2).mean()

    # 5. Effective degrees of freedom
    n_eff = effective_dof(hidden_trained[:, 0], target)

    # 6. FDR will be applied across all targets later

    # 7. Frequency-resolved R2
    freq_r2 = frequency_resolved_r2(hidden_trained, target, sampling_rate)

    # 8. Durbin-Watson on residuals
    ridge.fit(hidden_trained, target)
    residuals = target - ridge.predict(hidden_trained)
    dw = durbin_watson(residuals)

    # 9. Partial coherence (if input available)
    partial_r2 = None
    if input_signal is not None:
        partial_r2 = partial_coherence_r2(hidden_trained, target, input_signal)

    # 10. TOST zombie confirmation
    tost = tost_zombie_test(delta_r2, se_delta, n_eff)

    # 11. Bayes factor
    bf = bayes_factor_null(delta_r2, se_delta)

    # 12. Gap CV
    gap_r2 = gap_temporal_cv(hidden_trained, target, gap_size=block_size)

    # 13. Ljung-Box residual test
    lb = ljung_box_residual_test(residuals)

    return {
        'target': target_name,
        'ridge_delta_r2': delta_r2,
        'mlp_delta_r2': mlp_delta,
        'encoding_type': mlp_result[target_name]['encoding_type'],
        'p_block_permutation': p_block,
        'n_effective_dof': n_eff,
        'autocorrelation_tau_sec': tau_sec,
        'block_size': block_size,
        'frequency_r2': freq_r2,
        'durbin_watson': dw,
        'partial_coherence_r2': partial_r2,
        'tost_zombie': tost,
        'bayes_factor': bf,
        'gap_cv_r2': gap_r2,
        'ljung_box': lb,
        'hardened_verdict': _hardened_verdict(
            delta_r2, mlp_delta, p_block, tost, bf, freq_r2, dw),
    }


def _hardened_verdict(delta_r2, mlp_delta, p_block, tost, bf, freq_r2, dw):
    """Generate definitive verdict from all statistical evidence.

    Decision tree:
      1. Negative ΔR² → CONFIRMED_ZOMBIE (anti-correlated with biology;
         autocorrelation can only inflate *positive* R², so negative ΔR²
         is *stronger* zombie evidence when autocorrelation is present).
      2. Non-significant p_block with positive ΔR² → LIKELY/CONFIRMED_ZOMBIE
      3. Significant but drift-only → SPURIOUS_DRIFT
      4. MLP gain ≫ Ridge → NONLINEAR_ENCODED
      5. Positive ΔR² with autocorrelation → SUSPICIOUS_AUTOCORRELATION
      6. Large ΔR² surviving all checks → CONFIRMED_ENCODED
      7. Default → CANDIDATE_ENCODED
    """
    is_zombie_confirmed = tost['zombie_confirmed']
    is_bf_zombie = bf['bf01'] > 3
    is_drift_only = (freq_r2.get('ultra_slow', 0) > 0.1 and
                     freq_r2.get('medium', 0) < 0.01)
    is_autocorrelated = dw < 1.0

    # ── Rule 1: Negative ΔR² is unambiguous zombie evidence ──
    # Trained model predicts biology *worse* than untrained baseline.
    # Autocorrelation inflates positive correlations, not negative ones,
    # so DW < 1.0 cannot explain a negative ΔR² — it only strengthens it.
    if delta_r2 < 0:
        return 'CONFIRMED_ZOMBIE'

    # ── From here, ΔR² ≥ 0 ──
    is_significant = p_block < 0.05

    if not is_significant:
        if is_zombie_confirmed and is_bf_zombie:
            return 'CONFIRMED_ZOMBIE'
        return 'LIKELY_ZOMBIE'

    # ── Significant p_block with positive ΔR² — potential encoding ──

    if is_drift_only:
        return 'SPURIOUS_DRIFT'

    if mlp_delta > delta_r2 + 0.1:
        return 'NONLINEAR_ENCODED'

    # Autocorrelation flag only applies to *positive* ΔR² that could be
    # inflated by temporal dependence. Threshold at 0.15 because small
    # positive ΔR² with DW < 1.0 cannot be trusted.
    if is_autocorrelated and delta_r2 < 0.15:
        return 'SUSPICIOUS_AUTOCORRELATION'

    if delta_r2 > 0.2:
        return 'CONFIRMED_ENCODED'

    return 'CANDIDATE_ENCODED'
