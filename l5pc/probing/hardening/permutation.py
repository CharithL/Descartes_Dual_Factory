"""
L5PC DESCARTES -- Null distribution generators for statistical hardening.

Methods 1-3 from the 13-method suite:
  1. Block permutation (preserves local autocorrelation)
  2. Phase-randomized surrogates (IAAFT, preserves power spectrum)
  3. Circular shift null (Theiler 1986, preserves both autocorrelation structures)
"""

import numpy as np
from scipy.signal import correlate
from sklearn.linear_model import Ridge


def block_permute(y, block_size, rng):
    """
    Block-permute time series preserving local autocorrelation.
    Pads to next multiple of block_size to eliminate boundary bias.
    """
    T = y.shape[0]
    is_2d = y.ndim == 2

    pad_len = (block_size - T % block_size) % block_size
    if is_2d:
        y_padded = np.concatenate([y, np.zeros((pad_len, y.shape[1]))], axis=0)
    else:
        y_padded = np.concatenate([y, np.zeros(pad_len)])

    T_padded = y_padded.shape[0]
    n_blocks = T_padded // block_size

    if is_2d:
        blocks = y_padded.reshape(n_blocks, block_size, y.shape[1])
    else:
        blocks = y_padded.reshape(n_blocks, block_size)

    perm = rng.permutation(n_blocks)
    blocks_shuffled = blocks[perm]

    if is_2d:
        return blocks_shuffled.reshape(-1, y.shape[1])[:T]
    else:
        return blocks_shuffled.reshape(-1)[:T]


def adaptive_block_size(target, sampling_rate_hz=1000):
    """
    Compute block size from target autocorrelation.
    block_size = max(50, 3 * tau) where tau = 1/e crossing.
    """
    if target.ndim > 1:
        target = target[:, 0]

    target_centered = target - target.mean()
    acf = correlate(target_centered, target_centered, mode='full')
    acf = acf[len(acf)//2:]  # Positive lags only
    acf = acf / acf[0]       # Normalize

    # Find 1/e crossing
    threshold = 1.0 / np.e
    crossings = np.where(acf < threshold)[0]
    tau_samples = crossings[0] if len(crossings) > 0 else len(acf)

    block_size = max(50, int(3 * tau_samples))
    return block_size, tau_samples / sampling_rate_hz


def phase_randomize(signal, rng):
    """
    Phase-randomized surrogate preserving power spectrum exactly.
    Uses IAAFT (Iterative Amplitude Adjusted Fourier Transform).
    """
    n = len(signal)

    # FFT
    fft_vals = np.fft.rfft(signal)
    amplitudes = np.abs(fft_vals)

    # Randomize phases (preserve DC and Nyquist)
    random_phases = rng.uniform(0, 2*np.pi, size=len(fft_vals))
    random_phases[0] = 0  # DC
    if n % 2 == 0:
        random_phases[-1] = 0  # Nyquist

    # Reconstruct with random phases
    new_fft = amplitudes * np.exp(1j * random_phases)
    surrogate = np.fft.irfft(new_fft, n=n)

    # IAAFT: iteratively adjust to match both spectrum and amplitude distribution
    sorted_original = np.sort(signal)
    for _ in range(20):
        # Match amplitude distribution
        rank_order = np.argsort(np.argsort(surrogate))
        surrogate = sorted_original[rank_order]

        # Match power spectrum
        fft_surr = np.fft.rfft(surrogate)
        phases_surr = np.angle(fft_surr)
        fft_surr = amplitudes * np.exp(1j * phases_surr)
        surrogate = np.fft.irfft(fft_surr, n=n)

    return surrogate


def circular_shift_null(hidden, target, n_shifts=1000,
                         min_shift_factor=5, rng=None):
    """
    Circular shift null: shift target by > 5x max autocorrelation time.
    Preserves both autocorrelation structures while breaking alignment.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T = len(target)
    _, tau_sec = adaptive_block_size(target)
    min_shift = max(100, int(min_shift_factor * tau_sec * 1000))

    null_r2s = []
    ridge = Ridge(alpha=1.0)

    for _ in range(n_shifts):
        shift = rng.integers(min_shift, T - min_shift)
        target_shifted = np.roll(target, shift, axis=0)

        ridge.fit(hidden, target_shifted)
        null_r2s.append(ridge.score(hidden, target_shifted))

    return np.array(null_r2s)
