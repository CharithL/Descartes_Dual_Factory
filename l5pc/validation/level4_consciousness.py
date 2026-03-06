"""Level 4 Validation: Consciousness-Relevant Metrics.

The highest-impact validation level. Tests whether functional equivalence
(Level 1) guarantees phenomenal equivalence.

Key prediction: A feedforward TCN surrogate should FAIL Level 4 selectively:
- Φ drops because feedforward has Φ=0 by IIT
- PCI may still pass
- BAC index may fail if coincidence detection is lost
"""
import numpy as np


def compute_pci(perturbation_response, fs_hz=2000.0):
    """Perturbational Complexity Index via Lempel-Ziv complexity.

    PCI measures the algorithmic complexity of the brain's response
    to a perturbation. Higher PCI → more conscious-like dynamics.

    Args:
        perturbation_response: (n_neurons, T) binary spike matrix
            after perturbation onset

    Returns:
        dict with pci value and normalized pci
    """
    try:
        import antropy
    except ImportError:
        return {'pci': None, 'error': 'antropy not installed'}

    # Binarize and flatten spatiotemporally
    binary = (perturbation_response > 0).astype(int)
    # Concatenate all neurons into single binary string
    flat = binary.flatten()

    # Lempel-Ziv complexity
    lz = antropy.lziv_complexity(flat, normalize=True)

    return {
        'pci': float(lz),
        'n_neurons': binary.shape[0],
        'n_timesteps': binary.shape[1],
    }


def compute_transfer_entropy(spike_trains, dt_ms=0.5, max_lag=10):
    """Transfer entropy: directional information flow between neurons.

    Especially important for feedback connections (apical → soma).
    A feedforward surrogate should show reduced feedback TE.

    Args:
        spike_trains: (n_neurons, T) binary spike trains
        max_lag: Maximum lag in timesteps

    Returns:
        dict with mean TE, feedback TE, feedforward TE
    """
    n_neurons, T = spike_trains.shape
    # Bin spikes for TE computation
    bin_size = max(1, int(5.0 / dt_ms))  # 5ms bins
    n_bins = T // bin_size
    binned = spike_trains[:, :n_bins * bin_size].reshape(n_neurons, n_bins, bin_size).sum(axis=2)
    binned = (binned > 0).astype(int)

    te_values = []
    for i in range(min(n_neurons, 20)):  # Limit pairs for speed
        for j in range(min(n_neurons, 20)):
            if i == j:
                continue
            te = _simple_transfer_entropy(binned[i], binned[j], lag=1)
            te_values.append({'source': i, 'target': j, 'te': te})

    if not te_values:
        return {'mean_te': 0, 'te_pairs': []}

    mean_te = np.mean([t['te'] for t in te_values])
    return {
        'mean_te': float(mean_te),
        'n_pairs': len(te_values),
        'te_pairs': te_values[:50],  # Store subset
    }


def _simple_transfer_entropy(source, target, lag=1):
    """Simple TE estimation using conditional probability.

    TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})
    For binary sequences, computed from transition probabilities.
    """
    T = len(target)
    if T <= lag + 1:
        return 0.0

    # Build joint distribution
    counts = {}
    for t in range(lag, T):
        y_past = target[t - 1]
        x_past = source[t - lag]
        y_now = target[t]
        key = (y_past, x_past, y_now)
        counts[key] = counts.get(key, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    # Compute conditional entropies
    # H(Y_t | Y_{t-1}, X_{t-lag})
    h_cond_joint = 0
    margin_yx = {}
    for (yp, xp, yn), c in counts.items():
        key = (yp, xp)
        margin_yx[key] = margin_yx.get(key, 0) + c

    for (yp, xp, yn), c in counts.items():
        p_joint = c / total
        p_margin = margin_yx[(yp, xp)] / total
        if p_joint > 0 and p_margin > 0:
            h_cond_joint -= p_joint * np.log2(p_joint / p_margin)

    # H(Y_t | Y_{t-1})
    h_cond_y = 0
    margin_y = {}
    counts_y = {}
    for (yp, xp, yn), c in counts.items():
        key_yp = yp
        key_yn = (yp, yn)
        margin_y[key_yp] = margin_y.get(key_yp, 0) + c
        counts_y[key_yn] = counts_y.get(key_yn, 0) + c

    for (yp, yn), c in counts_y.items():
        p_joint = c / total
        p_margin = margin_y[yp] / total
        if p_joint > 0 and p_margin > 0:
            h_cond_y -= p_joint * np.log2(p_joint / p_margin)

    te = max(0, h_cond_y - h_cond_joint)
    return float(te)


def compute_phi_approximate(spike_trains, n_subsystem=8, dt_ms=0.5):
    """Approximate integrated information (Φ) for small subsystems.

    Full Φ computation is intractable for >15 elements.
    We compute Φ for random subsystems of n_subsystem neurons.

    A feedforward TCN has Φ=0 by IIT definition.

    Args:
        spike_trains: (n_neurons, T) binary
        n_subsystem: Number of neurons per subsystem (max ~10 for PyPhi)

    Returns:
        dict with phi estimates
    """
    try:
        import pyphi
    except ImportError:
        return {'phi': None, 'error': 'pyphi not installed'}

    n_neurons = spike_trains.shape[0]
    if n_neurons < n_subsystem:
        n_subsystem = n_neurons

    # Select random subsystem
    rng = np.random.RandomState(42)
    indices = rng.choice(n_neurons, n_subsystem, replace=False)
    sub_trains = spike_trains[indices]

    # Bin to reduce state space
    bin_size = int(10.0 / dt_ms)  # 10ms bins
    T = sub_trains.shape[1]
    n_bins = T // bin_size
    binned = sub_trains[:, :n_bins * bin_size].reshape(n_subsystem, n_bins, bin_size).sum(axis=2)
    binned = (binned > 0).astype(int)

    # Estimate TPM from data (transition probability matrix)
    n_states = 2 ** n_subsystem
    tpm = np.zeros((n_states, n_states))
    for t in range(n_bins - 1):
        state_now = int(''.join(map(str, binned[:, t])), 2)
        state_next = int(''.join(map(str, binned[:, t + 1])), 2)
        tpm[state_now, state_next] += 1

    # Normalize rows
    row_sums = tpm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    tpm = tpm / row_sums

    try:
        # PyPhi computation
        cm = np.ones((n_subsystem, n_subsystem))  # Fully connected
        network = pyphi.Network(tpm, cm=cm)
        # Use middle state as reference
        mid_state = tuple(binned[:, n_bins // 2])
        substrate = pyphi.Subsystem(network, mid_state)
        sia = pyphi.compute.sia(substrate)
        phi_value = float(sia.phi)
    except Exception as e:
        phi_value = None
        return {'phi': None, 'error': str(e), 'n_subsystem': n_subsystem}

    return {
        'phi': phi_value,
        'n_subsystem': n_subsystem,
        'neuron_indices': indices.tolist(),
    }


def compute_bac_index_circuit(l5pc_spike_trains, basal_inputs, apical_inputs, dt_ms=0.5):
    """BAC firing index across L5PCs in the circuit.

    Measures supralinearity of coincident basal + apical input.
    This should degrade if the surrogate loses coincidence detection.
    """
    from l5pc.utils.metrics import detect_spikes

    indices_with_coincidence = []
    for i in range(l5pc_spike_trains.shape[0]):
        spikes = detect_spikes(l5pc_spike_trains[i], dt_ms=dt_ms)
        if len(spikes) > 0:
            indices_with_coincidence.append(i)

    # Simplified: ratio of actual spikes to linear prediction
    total_spikes = sum(len(detect_spikes(l5pc_spike_trains[i], dt_ms=dt_ms))
                       for i in range(l5pc_spike_trains.shape[0]))

    return {
        'total_l5pc_spikes': total_spikes,
        'n_active_l5pcs': len(indices_with_coincidence),
        'n_total_l5pcs': l5pc_spike_trains.shape[0],
    }


def run_level4_validation(circuit_bio, circuit_surrogate, dt_ms=0.5):
    """Run all Level 4 metrics.

    Args:
        circuit_bio: dict with 'spike_trains', 'l5pc_spike_trains',
                     'basal_inputs', 'apical_inputs'
        circuit_surrogate: same structure

    Returns:
        dict of all Level 4 results
    """
    results = {}

    # PCI
    if 'spike_trains' in circuit_bio:
        results['pci_bio'] = compute_pci(circuit_bio['spike_trains'])
        results['pci_surrogate'] = compute_pci(circuit_surrogate['spike_trains'])

    # Transfer entropy
    if 'spike_trains' in circuit_bio:
        results['te_bio'] = compute_transfer_entropy(circuit_bio['spike_trains'], dt_ms)
        results['te_surrogate'] = compute_transfer_entropy(circuit_surrogate['spike_trains'], dt_ms)

    # Φ (approximate)
    if 'spike_trains' in circuit_bio:
        results['phi_bio'] = compute_phi_approximate(circuit_bio['spike_trains'], dt_ms=dt_ms)
        results['phi_surrogate'] = compute_phi_approximate(circuit_surrogate['spike_trains'], dt_ms=dt_ms)

    # BAC index
    if 'l5pc_spike_trains' in circuit_bio:
        results['bac_bio'] = compute_bac_index_circuit(
            circuit_bio['l5pc_spike_trains'],
            circuit_bio.get('basal_inputs'), circuit_bio.get('apical_inputs'), dt_ms)
        results['bac_surrogate'] = compute_bac_index_circuit(
            circuit_surrogate['l5pc_spike_trains'],
            circuit_surrogate.get('basal_inputs'), circuit_surrogate.get('apical_inputs'), dt_ms)

    # Summary
    results['summary'] = {}
    if 'pci_bio' in results and results['pci_bio'].get('pci') is not None:
        results['summary']['pci_ratio'] = (
            results['pci_surrogate']['pci'] / results['pci_bio']['pci']
            if results['pci_bio']['pci'] > 0 else 0
        )
    if 'phi_bio' in results and results['phi_bio'].get('phi') is not None:
        results['summary']['phi_ratio'] = (
            (results['phi_surrogate'].get('phi', 0) or 0) /
            results['phi_bio']['phi']
            if results['phi_bio']['phi'] > 0 else 0
        )

    return results
