"""
L5PC DESCARTES -- Multiple comparison corrections and zombie confirmation.

Methods 5, 9, 10 from the 13-method suite:
  5. FDR correction (Benjamini-Hochberg)
  9. TOST equivalence testing (zombie confirmation)
  10. Bayes factor for null (Savage-Dickey)
"""

import numpy as np
from scipy.stats import t as t_dist, norm, halfnorm


def fdr_correction(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction.
    Returns adjusted p-values and significance mask.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    adjusted = np.zeros(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]

    for i in range(n-2, -1, -1):
        adjusted[sorted_idx[i]] = min(
            adjusted[sorted_idx[i+1]],
            sorted_p[i] * n / (i + 1)
        )

    return adjusted, adjusted < alpha


def tost_zombie_test(delta_r2, se_delta_r2, n_eff,
                      equivalence_bound=0.05):
    """
    Two One-Sided Tests for confirming zombie null.
    H0: |delta-R2| >= bound (non-zombie)
    H1: |delta-R2| < bound (zombie confirmed)

    Returns p-value for zombie confirmation.
    """
    df = n_eff - 2

    # Test 1: delta-R2 > -bound
    t1 = (delta_r2 - (-equivalence_bound)) / se_delta_r2
    p1 = t_dist.cdf(t1, df)  # Should be large

    # Test 2: delta-R2 < +bound
    t2 = (delta_r2 - equivalence_bound) / se_delta_r2
    p2 = 1 - t_dist.cdf(t2, df)  # Should be large

    # TOST p-value = max of the two
    p_tost = max(1 - p1, p2)

    return {
        'p_tost': p_tost,
        'zombie_confirmed': p_tost < 0.05,
        'delta_r2': delta_r2,
        'equivalence_bound': equivalence_bound,
    }


def bayes_factor_null(delta_r2, se_delta_r2, prior_scale=0.1):
    """
    Bayes factor BF01: evidence for null (zombie) vs alternative.
    BF01 > 3 -> substantial evidence for zombie.
    BF01 > 10 -> strong evidence.

    Uses Savage-Dickey density ratio with half-normal prior.
    """
    # Posterior density at 0
    posterior_at_0 = norm.pdf(0, loc=delta_r2, scale=se_delta_r2)

    # Prior density at 0 (half-normal with scale = prior_scale)
    prior_at_0 = halfnorm.pdf(0, scale=prior_scale)

    bf01 = posterior_at_0 / (prior_at_0 + 1e-10)

    return {
        'bf01': bf01,
        'interpretation': (
            'STRONG_ZOMBIE' if bf01 > 10 else
            'MODERATE_ZOMBIE' if bf01 > 3 else
            'INCONCLUSIVE' if bf01 > 1/3 else
            'MODERATE_NON_ZOMBIE' if bf01 > 1/10 else
            'STRONG_NON_ZOMBIE'
        ),
    }
