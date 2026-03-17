"""
L5PC DESCARTES -- Tier 3: Dynamical Probes

Koopman spectral analysis, SINDy symbolic regression, DSA delay-embedded comparison.

Tests whether hidden state *dynamics* match biological dynamics —
same eigenvalues, same equations, same dynamical manifold.
"""

import logging

import numpy as np

from l5pc.probing.registry import is_available

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 7.1  Koopman Spectral Analysis
# ──────────────────────────────────────────────────────────────

def koopman_spectral_comparison(hidden_trajectories, bio_trajectories,
                                  n_modes=20, dt=0.001):
    """
    Koopman operator: compare eigenvalue spectra of learned vs biological dynamics.

    Matching eigenvalues -> matching timescales and oscillation frequencies.
    Matching eigenvectors -> matching dynamical modes.
    """
    from scipy.linalg import eig

    def estimate_koopman(trajectories, n_modes):
        """DMD-based Koopman estimation."""
        # Stack time-shifted pairs: X = [x(0), ..., x(T-1)], Y = [x(1), ..., x(T)]
        X = np.concatenate([t[:-1] for t in trajectories], axis=0)
        Y = np.concatenate([t[1:] for t in trajectories], axis=0)

        # DMD: K = Y @ pinv(X)
        U, s, Vt = np.linalg.svd(X.T, full_matrices=False)
        U = U[:, :n_modes]
        s = s[:n_modes]
        Vt = Vt[:n_modes, :]

        K_tilde = U.T @ Y.T @ Vt.T @ np.diag(1.0 / s)
        eigenvalues, eigenvectors = eig(K_tilde)

        # Convert to continuous-time
        lambdas = np.log(eigenvalues + 1e-10) / dt
        frequencies = np.abs(np.imag(lambdas)) / (2 * np.pi)
        decay_rates = np.real(lambdas)

        return eigenvalues, frequencies, decay_rates, eigenvectors

    eig_h, freq_h, decay_h, modes_h = estimate_koopman(hidden_trajectories, n_modes)
    eig_b, freq_b, decay_b, modes_b = estimate_koopman(bio_trajectories, n_modes)

    # Compare: sort by magnitude and compute distance
    idx_h = np.argsort(-np.abs(eig_h))
    idx_b = np.argsort(-np.abs(eig_b))

    freq_match = np.corrcoef(
        np.sort(freq_h[idx_h[:n_modes]]),
        np.sort(freq_b[idx_b[:n_modes]]))[0, 1]

    decay_match = np.corrcoef(
        np.sort(decay_h[idx_h[:n_modes]]),
        np.sort(decay_b[idx_b[:n_modes]]))[0, 1]

    return {
        'frequency_correlation': float(freq_match),
        'decay_rate_correlation': float(decay_match),
        'hidden_frequencies': freq_h[idx_h[:10]].tolist(),
        'bio_frequencies': freq_b[idx_b[:10]].tolist(),
        'spectral_match': freq_match > 0.7 and decay_match > 0.5
    }


# ──────────────────────────────────────────────────────────────
# 7.2  SINDy Symbolic Regression
# ──────────────────────────────────────────────────────────────

def sindy_probe(hidden_trajectories, dt=0.001, poly_order=3,
                 threshold=0.05):
    """
    SINDy (Sparse Identification of Nonlinear Dynamics):
    Discover governing equations from hidden state trajectories.

    If discovered equations resemble HH: dh/dt = alpha(1-h) - beta*h,
    the network literally rediscovered the biological dynamics.

    Requires: pip install pysindy
    """
    if not is_available('sindy'):
        return {'error': 'pysindy not installed'}

    try:
        import pysindy as ps

        # Stack trajectories
        X = np.concatenate(hidden_trajectories, axis=0)

        # Fit SINDy model
        model = ps.SINDy(
            feature_library=ps.PolynomialLibrary(degree=poly_order),
            optimizer=ps.STLSQ(threshold=threshold),
        )
        model.fit(X, t=dt)

        # Extract equations
        equations = model.equations()
        coefficients = model.coefficients()

        # Sparsity: fraction of nonzero coefficients
        sparsity = (np.abs(coefficients) > threshold).mean()

        # Score: prediction accuracy
        X_dot_pred = model.predict(X)
        X_dot_true = np.gradient(X, dt, axis=0)
        r2 = 1 - np.sum((X_dot_true - X_dot_pred)**2) / (
            np.sum((X_dot_true - X_dot_true.mean(0))**2) + 1e-10)

        return {
            'equations': equations,
            'n_terms': int((np.abs(coefficients) > threshold).sum()),
            'sparsity': float(sparsity),
            'prediction_r2': float(r2),
            'hh_similarity': _compare_to_hh(coefficients, poly_order)
        }
    except ImportError:
        return {'error': 'pysindy not installed'}


def _compare_to_hh(coefficients, poly_order):
    """
    Compare discovered equations to HH structure.
    HH gating: dX/dt = alpha(V)(1-X) - beta(V)X = alpha - (alpha+beta)X
    This is a first-order polynomial in X with voltage-dependent coefficients.
    Score based on structural similarity.
    """
    # Check if equations are approximately linear in hidden state
    # (nonzero linear terms, small quadratic+ terms)
    n_dims = coefficients.shape[0]

    if poly_order >= 2:
        n_linear = coefficients.shape[1] - 1  # Exclude constant
        linear_energy = np.abs(coefficients[:, 1:n_dims + 1]).sum()
        total_energy = np.abs(coefficients[:, 1:]).sum() + 1e-10
        linearity_ratio = linear_energy / total_energy
    else:
        linearity_ratio = 1.0

    return float(linearity_ratio)


# ──────────────────────────────────────────────────────────────
# 7.3  DSA (Dynamical Similarity Analysis)
# ──────────────────────────────────────────────────────────────

def dsa_comparison(hidden_trajectories, bio_trajectories,
                    delay=1, rank=20):
    """
    DSA (Ostrow et al. 2023): Compare dynamics using
    delay-embedded state-transition matrices.

    More robust than Koopman for nonlinear systems.
    """
    def delay_embed(trajs, delay, rank):
        embedded = []
        for traj in trajs:
            T, D = traj.shape
            if T <= delay:
                continue
            X = np.concatenate([traj[i:T - delay + i] for i in range(delay + 1)], axis=1)
            embedded.append(X)
        return np.concatenate(embedded, axis=0)

    X_h = delay_embed(hidden_trajectories, delay, rank)
    X_b = delay_embed(bio_trajectories, delay, rank)

    # SVD truncation to shared rank
    U_h, s_h, _ = np.linalg.svd(X_h, full_matrices=False)
    U_b, s_b, _ = np.linalg.svd(X_b, full_matrices=False)

    U_h = U_h[:, :rank]
    U_b = U_b[:, :rank]

    # Principal angle between subspaces
    _, sigma, _ = np.linalg.svd(U_h.T @ U_b)
    principal_angles = np.arccos(np.clip(sigma[:rank], -1, 1))

    dsa_distance = np.mean(principal_angles)

    return {
        'dsa_distance': float(dsa_distance),
        'principal_angles': principal_angles.tolist(),
        'dynamical_match': dsa_distance < np.pi / 4  # < 45 degrees
    }
