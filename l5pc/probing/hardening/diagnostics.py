"""
L5PC DESCARTES -- Residual diagnostics for probing validation.

Methods 4, 7, 13 from the 13-method suite:
  4. Effective degrees of freedom (Bartlett's formula)
  7. Durbin-Watson autocorrelation diagnostic
  13. Ljung-Box residual autocorrelation test
"""

import numpy as np
from scipy.signal import correlate
from scipy.stats import chi2


def effective_dof(x, y):
    """
    Bartlett's formula: effective N for autocorrelated series.
    N_eff = N / (1 + 2 * sum(acf_x(k) * acf_y(k)))
    """
    N = len(x)
    x_c = (x - x.mean()) / (x.std() + 1e-10)
    y_c = (y - y.mean()) / (y.std() + 1e-10)

    acf_x = correlate(x_c, x_c, mode='full')[N-1:] / N
    acf_y = correlate(y_c, y_c, mode='full')[N-1:] / N

    max_lag = min(N // 3, 500)
    correction = 1 + 2 * np.sum(acf_x[1:max_lag] * acf_y[1:max_lag])

    n_eff = max(3, N / correction)
    return n_eff


def durbin_watson(residuals):
    """
    DW statistic: DW << 2 -> positive autocorrelation -> spurious regression risk.
    DW < 1.0 is a red flag. DW near 2.0 is acceptable.
    """
    diff = np.diff(residuals, axis=0)
    return float(np.sum(diff**2) / np.sum(residuals**2))


def ljung_box_residual_test(residuals, n_lags=20):
    """
    Test whether Ridge probe residuals are autocorrelated.
    Significant result -> model is missing temporal structure.
    """
    n = len(residuals)
    acf = np.correlate(residuals - residuals.mean(),
                        residuals - residuals.mean(), 'full')
    acf = acf[n-1:] / acf[n-1]

    Q = n * (n + 2) * sum(acf[k]**2 / (n - k) for k in range(1, n_lags + 1))
    p_value = 1 - chi2.cdf(Q, n_lags)

    return {'Q_statistic': Q, 'p_value': p_value,
            'autocorrelated': p_value < 0.05}
