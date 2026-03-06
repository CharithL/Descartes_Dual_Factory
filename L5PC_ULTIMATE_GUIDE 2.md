# L5PC DESCARTES Ultimate Guide: The Cortical Zombie Test

## From Hippocampal Proof-of-Concept to Cortical Validation

**Version 1.0 — March 2026**
**ARIA COGITO Programme | Computational Neuroscience**

---

## 0. What This Guide Is

This is the definitive implementation guide for extending the DESCARTES zombie test from the hippocampal CA3→CA1 circuit to the Layer 5 thick-tufted pyramidal cell (L5PC). It integrates three prior design documents — the identifiability analysis, the four-level validation framework, and the full biological state recording blueprint — with the critical methodological lessons learned from the hippocampal bottleneck sweep experiment.

The hippocampal experiment established four findings that reshape the L5PC design:

1. **gamma_amp is the sole mandatory variable** out of 25 — the LSTM independently discovers it because the CA3→CA1 transformation mathematically requires tracking oscillatory timing. All other biological variables are zombie.

2. **Two types of mandatory variable exist**: concentrated (g_NMDA_SC, breaks at 4% clamp) and holistic (gamma_amp, breaks at 60%+ clamp). The L5PC experiment must test for both.

3. **Individual gating variables are expected to be zombie** — this is not a failure but a mathematical consequence of structural non-identifiability (Walch & Eisenberg 2016). The correct zombie test probes effective conductances, ionic currents, and emergent dendritic properties.

4. **Causal ablation with progressive clamping is the decisive test** — ΔR² alone cannot distinguish a variable that is learned-and-used from one that is learned-and-ignored. The L5PC guide must include the full ablation protocol from the start.

---

## 1. Why L5PC Is the Right Next Circuit

Three circuits form a hierarchy of computational complexity:

| Circuit | Topology | Bio vars | Scale | Key computation |
|---------|----------|----------|-------|-----------------|
| Thalamus (Le Masson) | 2-cell feedback loop | 160 | Network-level | Tonic/oscillatory bifurcation |
| Hippocampus (MIMO) | Population feedforward | 25 | Population-level | Rate transformation |
| **L5 Pyramidal Cell** | **Single neuron, 639 compartments** | **~5,000 (full) / ~90 (Bahl)** | **Within-neuron** | **Dendritic computation** |

The L5PC tests something the other circuits cannot: whether computation *within a single neuron* has mandatory intermediates. The Hay 2011 model features calcium spikes in the apical tuft, NMDA plateau potentials, back-propagating action potentials, and a calcium hot zone at 685–885 μm from the soma. These are computations every cortical pyramidal neuron performs. If a surrogate discovers dendritic calcium dynamics as mandatory — the way the hippocampal LSTM discovered gamma_amp — that finding applies to every neuron in the cortex.

Beniaguev et al. (2021) trained a TCN on the full Hay model and achieved 99.1% spike prediction accuracy (AUC = 0.9911). But they never probed the hidden states. They had the ground truth, they had the trained model, and they stopped at "it works." This guide completes what they started.

---

## 2. The Identifiability Correction

### 2.1 Why Individual Gates Are the Wrong Target

Walch & Eisenberg (2016) proved that individual steady-state gating variables (m, h, n) are structurally non-identifiable from voltage-clamp data. Only their products with maximal conductance form identifiable combinations. The proof is a scaling argument: replacing m∞ → α·m∞ and ḡ → ḡ/αᵖ leaves measured current invariant, creating infinite observationally equivalent parameter sets.

This means R² ≈ 0 for individual gates is the mathematically expected result for any correctly functioning approximator, not evidence of zombie-ness. The A-R3 thalamic finding of R² ≈ 0 for 160 individual gating variables must be reclassified from "evidence of zombie" to "evidence of probing for non-identifiable quantities."

### 2.2 The Three-Level Probing Hierarchy

The L5PC zombie test probes at three levels, from expected-zombie to potentially-non-zombie:

**Level A — Individual Gating Variables (expected zombie)**

| Variable type | Count (Bahl) | Count (Hay) | Expected ΔR² | Purpose |
|---------------|-------------|-------------|---------------|---------|
| m, h gates (voltage-dependent) | ~25 | ~2,500 | ≈ 0 | Confirm identifiability theory |
| z gate (Ca²⁺-dependent, SK_E2) | ~2 | ~76 | ≈ 0 (unless Hill normalization constrains) | Test Ca-gating edge case |
| [Ca²⁺]ᵢ (observable) | ~3 | ~200 | Potentially > 0 | Calcium is directly observable |

Finding ΔR² ≈ 0 at Level A is not a failure — it confirms the identifiability prediction and validates the probing methodology.

**Level B — Effective Conductances and Ionic Currents (the real zombie test)**

| Probing target | Formula | Count (Bahl) | Count (Hay) |
|----------------|---------|-------------|-------------|
| G_NaTa(t) | ḡ·m³·h | Per-compartment where present | ~200 |
| G_KTst(t) | ḡ·m⁴·h | Per-compartment | ~3 |
| G_CaHVA(t) | ḡ·m²·h | Per-compartment (hot zone enriched) | ~80 |
| G_CaLVA(t) | ḡ·m²·h | Per-compartment (hot zone enriched) | ~80 |
| G_Ih(t) | ḡ·m | Per-compartment (exponential gradient) | ~500 |
| G_Im(t) | ḡ·m | Apical dendrites | ~380 |
| G_SK(t) | ḡ·z | Per-compartment | ~80 |
| G_SKv3(t) | ḡ·m | Soma + some apical | ~100 |
| I_Na(t) total | Sum of Na channel currents | Per-compartment | ~639 |
| I_K(t) total | Sum of K channel currents | Per-compartment | ~639 |
| I_Ca(t) total | Sum of Ca channel currents | Per-compartment | ~200 |
| I_h(t) | HCN current | Per-compartment | ~500 |

For the Bahl reduced model, aggregate by region: soma (1 compartment), basal (1), apical trunk (1), apical nexus/hot zone (1), tuft (1-2). This yields approximately **55 effective conductance/current targets**.

Ionic currents are likely the best probing targets because: (a) they are what the cable equation directly "sees," (b) they compress information into fewer quantities, and (c) Burghi et al. (2025) showed RMMs naturally recover currents.

**Level C — Emergent Dendritic Properties (the gamma_amp equivalents)**

These are the L5PC's mandatory variable candidates — higher-order properties arising from multi-channel, multi-compartment interactions:

| Property | How to compute | Biological role | Prediction |
|----------|---------------|-----------------|------------|
| BAC firing index | Supralinearity ratio: burst output / (basal-only + apical-only outputs) | Coincidence detection | **Mandatory, holistic** (gamma-type) |
| Ca hot zone activation | Peak [Ca²⁺]ᵢ in 685–885 μm region | Dendritic calcium spike | **Mandatory, holistic** |
| Burst/tonic ratio | Fraction of spikes occurring in bursts (ISI < 10ms) | Mode switching | **Mandatory, concentrated** |
| Dendritic Ca spike amplitude | Peak V in apical tuft during calcium spike | Apical integration | Potentially mandatory |
| Critical frequency | Minimum somatic firing rate triggering dendritic Ca spike | Frequency-dependent gain | Potentially mandatory |
| Apical-basal coincidence window | Cross-correlation peak between apical and basal EPSP timing | Temporal integration | Potentially mandatory |
| Total excitatory drive (per region) | Sum g_AMPA + g_NMDA per dendritic region | Regional input balance | Likely zombie (directly available from input) |
| Total inhibitory drive | Sum g_GABA per region | E/I balance | Likely zombie |
| Somatic I_Na peak | Maximum sodium current during AP | Spike generation | Likely NMDA-type (concentrated) |
| AHP depth | Post-spike hyperpolarization amplitude | Firing rate adaptation | Unknown |

### 2.3 The Voltage-Only Baseline Control

Because effective conductances are voltage-dependent at steady state (G(V) = ḡ·m∞(V)ᵖ·h∞(V)ᵍ), a network that merely re-encodes voltage would show high R² for effective conductances trivially. Three baselines are required:

1. **Untrained network baseline**: Random initialisation hidden states probed for the same targets. ΔR² = R²_trained − R²_untrained is the primary metric (identical to hippocampal methodology).

2. **Voltage-only baseline**: Compute R² between probing targets and a function of local voltage alone. If the trained network's R² does not exceed this, it has merely learned to encode voltage, not channel-specific dynamics.

3. **Temporal baseline**: Exponentially filtered voltage (matching each channel's τ) provides a stronger baseline. The network must exceed this to demonstrate genuine multi-timescale representation.

---

## 3. The Two-Phase Strategy: Bahl First, Then Hay

### 3.1 Why Bahl Before Hay

| Property | Bahl 2012 (reduced) | Hay 2011 (full) |
|----------|-------------------|-----------------|
| Compartments | ~6 | ~639 |
| Ion channel types | 9 | 12 |
| State variables | ~90 | ~5,000 |
| Preserves BAC firing | Yes | Yes |
| Preserves burst/tonic | Yes | Yes |
| Preserves calcium hot zone | Yes | Yes |
| Storage per trial (1ms) | ~0.2 MB | ~11.4 MB |
| 500 trials total | ~100 MB | ~5.7 GB |
| Simulation time (500 trials) | ~30 min | ~2 hours |
| Ridge probing time | ~1 hour | ~1 day |
| ModelDB accession | 146026 | 139653 |

The Bahl model has ~90 biological variables — comparable to the thalamic circuit (160) and hippocampal circuit (25). The full DESCARTES pipeline runs in a day. If Bahl results are interesting, scale to the full Hay model for the paper. If Bahl results are universally zombie even at Level B and C, the full Hay model won't change that conclusion.

### 3.2 Phase 1: Bahl Reduced Model (Weeks 1–3)

**Goal**: Complete zombie test with three-level probing, causal ablation, and progressive ablation on the 6-compartment L5PC.

**Deliverable**: Cross-circuit comparison table (thalamic vs hippocampal vs L5PC) with mandatory variable identification.

### 3.3 Phase 2: Full Hay Model (Weeks 4–8)

**Goal**: Scale to 639-compartment model with spatial resolution. Test whether mandatory variables found in Bahl are spatially localised or distributed across the dendritic tree.

**Deliverable**: Spatial map of mandatory variables overlaid on dendritic morphology. Publication-quality figures.

### 3.4 Phase 3: Circuit Integration (Weeks 9–14)

**Goal**: Embed the L5PC surrogate in a 100-neuron cortical microcircuit (Hay & Segev 2015 recurrent L5 circuit + PV+ basket cells). Perform graded replacement and four-level validation.

**Deliverable**: Replacement degradation curve with Level 1–4 metrics.

---

## 4. Phase 1 Implementation: Bahl Reduced Model

### 4.1 Setup

```bash
# 1. Install NEURON
pip install neuron

# 2. Download Bahl model from ModelDB 146026
# Alternatively: git clone from OpenSourceBrain
# Key files: soma, basal, apical trunk, nexus, tuft sections
# with NaTa_t, Ca_HVA, Ca_LVA, Ih, Im, SK, SKv3, K_Tst, K_Pst, pas

# 3. Compile mod files
cd bahl_model/mechanisms
nrnivmodl .
```

### 4.2 Stimulation Protocol

The stimulation must cover the full dynamic range including BAC firing events:

| Condition | Basal input rate | Apical input rate | Expected output | # Trials |
|-----------|-----------------|-------------------|-----------------|----------|
| Subthreshold | 1–3 Hz | 0–1 Hz | No spikes | 50 |
| Tonic firing | 5–15 Hz | 0–3 Hz | Regular spiking | 100 |
| Burst firing | 15–30 Hz | 10–20 Hz | Burst mode | 100 |
| BAC-specific | 5–10 Hz (basal) | 10–20 Hz (apical, simultaneous) | BAC spikes | 100 |
| Mixed/random | 0–30 Hz (random) | 0–20 Hz (random) | Variable | 150 |

**Total: 500 trials. Split: 350 train / 75 val / 75 test.**

**Critical design choice**: Basal and apical inputs must be delivered as SEPARATE input channels to the surrogate. If combined into a single input vector, the surrogate cannot learn coincidence detection because it cannot distinguish where on the dendritic tree input arrived. This is analogous to providing CA3 spikes as the hippocampal input — the spatial structure must be preserved.

```python
# Input format per trial:
# X_basal:  (T, n_basal_synapses)  — binary spike times for basal dendrite synapses
# X_apical: (T, n_apical_synapses) — binary spike times for apical tuft synapses
# X_soma:   (T, n_soma_synapses)   — inhibitory synapses (perisomatic)
# Concatenated: X = (T, n_basal + n_apical + n_soma)

# Output format per trial:
# Y: (T,) — somatic spike train (binary at 0.5ms or 1ms bins)
```

### 4.3 Recording All Biological Variables

Record at 0.5ms resolution (every 20th NEURON timestep at dt=0.025ms). This gives 2000 timepoints per 1000ms trial (or 1200 per 600ms trial).

**Level A — Individual Gates (~30 variables for Bahl)**

For each compartment where each mechanism is inserted:

```python
recordings = {}
for sec in h.allsec():
    for seg in sec:
        seg_name = f"{sec.name()}({seg.x:.4f})"
        recordings[seg_name] = {'v': h.Vector().record(seg._ref_v)}
        
        for mech, vars_list in [
            ('NaTa_t', ['m', 'h']), ('Nap_Et2', ['m', 'h']),
            ('K_Pst', ['m', 'h']), ('K_Tst', ['m', 'h']),
            ('SKv3_1', ['m']), ('SK_E2', ['z']),
            ('Im', ['m']), ('Ih', ['m']),
            ('Ca_HVA', ['m', 'h']), ('Ca_LVAst', ['m', 'h']),
        ]:
            if hasattr(seg, f'{vars_list[0]}_{mech}'):
                for var in vars_list:
                    key = f'{var}_{mech}'
                    recordings[seg_name][key] = h.Vector().record(
                        getattr(seg, f'_ref_{key}'))
        
        for cur in ['ina', 'ik', 'ica']:
            if hasattr(seg, cur):
                recordings[seg_name][cur] = h.Vector().record(
                    getattr(seg, f'_ref_{cur}'))
        if hasattr(seg, 'ihcn_Ih'):
            recordings[seg_name]['ihcn_Ih'] = h.Vector().record(seg._ref_ihcn_Ih)
        if hasattr(seg, 'cai'):
            recordings[seg_name]['cai'] = h.Vector().record(seg._ref_cai)
```

**Level B — Effective Conductances (computed post-hoc from Level A recordings)**

```python
def compute_effective_conductances(gates, gbar_values):
    """
    Compute identifiable combinations from recorded gates.
    
    For NaTa_t: G_eff = gbar_NaTa * m^3 * h
    For K_Tst:  G_eff = gbar_KTst * m^4 * h
    For Ca_HVA: G_eff = gbar_CaHVA * m^2 * h
    etc.
    """
    G_eff = {}
    
    # Voltage-gated channels with m^p * h^q gating
    channel_specs = {
        'NaTa_t':  {'gates': ['m', 'h'], 'exp': [3, 1]},
        'Nap_Et2': {'gates': ['m', 'h'], 'exp': [3, 1]},
        'K_Tst':   {'gates': ['m', 'h'], 'exp': [4, 1]},
        'K_Pst':   {'gates': ['m', 'h'], 'exp': [2, 1]},
        'SKv3_1':  {'gates': ['m'],      'exp': [1]},
        'SK_E2':   {'gates': ['z'],      'exp': [1]},
        'Im':      {'gates': ['m'],      'exp': [1]},
        'Ih':      {'gates': ['m'],      'exp': [1]},
        'Ca_HVA':  {'gates': ['m', 'h'], 'exp': [2, 1]},
        'Ca_LVAst':{'gates': ['m', 'h'], 'exp': [2, 1]},
    }
    
    for ch_name, spec in channel_specs.items():
        if ch_name in gbar_values:
            product = gbar_values[ch_name]
            for gate, exp in zip(spec['gates'], spec['exp']):
                gate_key = f'{gate}_{ch_name}'
                if gate_key in gates:
                    product = product * (gates[gate_key] ** exp)
            G_eff[f'G_{ch_name}'] = product
    
    return G_eff
```

Also compute ionic currents:
```python
def compute_ionic_currents(G_eff, V, reversal_potentials):
    """I_c(t) = G_c(t) * (V(t) - E_c)"""
    I = {}
    for ch_name, G in G_eff.items():
        E = reversal_potentials[ch_name]
        I[f'I_{ch_name}'] = G * (V - E)
    return I
```

**Level C — Emergent Properties (computed post-hoc)**

```python
def compute_emergent_properties(recordings, spike_times):
    """
    Compute higher-order dendritic properties — the gamma_amp equivalents.
    """
    props = {}
    
    # BAC firing index: supralinearity of coincident input
    # (requires separate basal-only and apical-only control trials)
    # BAC_index = n_spikes_combined / (n_spikes_basal_only + n_spikes_apical_only)
    
    # Calcium hot zone activation
    nexus_cai = recordings['apical_nexus']['cai']
    props['Ca_hotzone_peak'] = np.max(nexus_cai)  # per window
    props['Ca_hotzone_mean'] = np.mean(nexus_cai)
    
    # Burst/tonic ratio
    if len(spike_times) > 1:
        isis = np.diff(spike_times)
        burst_spikes = np.sum(isis < 10.0)  # ISI < 10ms = burst
        props['burst_ratio'] = burst_spikes / len(isis)
    else:
        props['burst_ratio'] = 0.0
    
    # Dendritic calcium spike amplitude
    tuft_v = recordings['tuft']['v']
    props['dendritic_Ca_amplitude'] = np.max(tuft_v) - np.mean(tuft_v[:100])
    
    # AHP depth
    soma_v = recordings['soma']['v']
    for st in spike_times:
        idx = int(st / dt)
        if idx + 50 < len(soma_v):
            props['AHP_depth'] = np.min(soma_v[idx:idx+50]) - soma_v[idx-1]
    
    # Somatic I_Na peak (per spike)
    props['I_Na_peak'] = np.min(recordings['soma']['ina'])  # Na current is negative
    
    return props
```

### 4.4 Save Format

Per trial, save to HDF5 or numpy:

```
trial_NNN/
  inputs.npy:         (T, n_synapses) — synaptic activation patterns
  output.npy:         (T,) — somatic spikes
  level_A_gates.npy:  (T, ~30) — individual gating variables
  level_B_cond.npy:   (T, ~25) — effective conductances per region
  level_B_curr.npy:   (T, ~25) — ionic currents per region
  level_C_emerge.npy: (T, ~12) — emergent properties per window
```

Variable name metadata:
```
variable_names.json: {
  "level_A": ["m_NaTa_soma", "h_NaTa_soma", "m_Ih_soma", ...],
  "level_B_cond": ["G_NaTa_soma", "G_CaHVA_nexus", "G_Ih_apical", ...],
  "level_B_curr": ["I_Na_soma", "I_Ca_nexus", "I_h_apical", ...],
  "level_C": ["Ca_hotzone_peak", "burst_ratio", "BAC_index", ...]
}
```

### 4.5 Train Surrogates

Two architectures, maintaining continuity with hippocampal results:

**Architecture 1: LSTM (cross-circuit comparison)**

```python
# Identical config to hippocampal sweep
hidden_sizes = [64, 128, 256]  # sweep
n_layers = 2
input_dim = n_basal_syn + n_apical_syn + n_soma_syn
output_dim = 1  # somatic spike probability
loss = MSE  # on firing rates, matching hippocampal methodology
optimizer = AdamW(lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(patience=10, factor=0.5)
early_stopping = patience=15
```

**Architecture 2: TCN (Beniaguev replication)**

```python
# Replicate Beniaguev's architecture
n_layers = 7
n_features = 128
temporal_receptive_field = 153  # ms
# Train from scratch on Bahl data (NOT pre-trained Hay weights)
# Reason: Bahl has different morphology, pre-trained weights won't transfer
```

For each trained model:
- Extract trained hidden states for all test trials
- Create untrained baseline with identical architecture, random init
- Extract untrained hidden states
- Save both to `sweep_hNNN/trained_hidden.npy` and `untrained_hidden.npy`

### 4.6 Ridge ΔR² Probing (Three Levels)

Use the IDENTICAL methodology from `bottleneck_ridge_all.py`:

```python
# For each level (A, B, C) and each target variable:
probe_config = {
    'cv': 5,  # trial-level folds
    'preprocessing': ['Raw', 'StandardScaler', 'PCA_5', 'PCA_10', 'PCA_20', 'PCA_50'],
    'decoder': RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000]),
    'selectivity_perms': 20,
    'p_threshold': 0.05,
}
```

Report three separate results tables:

```
LEVEL A — Individual Gates (Bahl, h=128 LSTM)
Variable              | R²_trained | R²_untrained | ΔR²    | Category
m_NaTa_soma           |   X.XXX    |    X.XXX     | X.XXX  | [ZOMBIE/LEARNED]
h_NaTa_soma           |   X.XXX    |    X.XXX     | X.XXX  | ...
...
Expected: mostly ΔR² ≈ 0 (confirms identifiability theory)

LEVEL B — Effective Conductances and Currents
G_NaTa_soma           |   X.XXX    |    X.XXX     | X.XXX  | [ZOMBIE/LEARNED]
G_CaHVA_nexus         |   X.XXX    |    X.XXX     | X.XXX  | ...
I_Na_soma             |   X.XXX    |    X.XXX     | X.XXX  | ...
I_Ca_nexus            |   X.XXX    |    X.XXX     | X.XXX  | ...
THIS IS THE KEY TABLE. Non-zombie here = genuine mechanistic encoding.

LEVEL C — Emergent Properties
Ca_hotzone_peak       |   X.XXX    |    X.XXX     | X.XXX  | [ZOMBIE/LEARNED]
burst_ratio           |   X.XXX    |    X.XXX     | X.XXX  | ...
BAC_index             |   X.XXX    |    X.XXX     | X.XXX  | ...
These are the L5PC's gamma_amp candidates.
```

### 4.7 Voltage-Only and Temporal Baselines

For every Level B target with ΔR² > 0.1, run two additional controls:

**Voltage-only baseline**: Fit Ridge regression from local compartment voltage V(t) to each effective conductance G(t). Since G∞(V) = ḡ·m∞(V)ᵖ·h∞(V)ᵍ, a high R² from voltage alone means the conductance is trivially determined by voltage.

```python
# For each effective conductance target:
R2_voltage_only = ridge_cv(V_compartment, G_target)
# The network must exceed this to claim genuine channel-specific encoding
R2_above_voltage = R2_trained - R2_voltage_only
```

**Temporal baseline**: Exponentially filtered voltage with time constant matching each channel's τ:

```python
# For each channel with known τ(V):
V_filtered = exponential_filter(V_compartment, tau=channel_tau_ms)
R2_temporal = ridge_cv(V_filtered, G_target)
# The network must exceed this to demonstrate multi-timescale representation
```

**Interpretation thresholds:**

| R² range | Interpretation |
|----------|---------------|
| ΔR² > 0.2 AND R²_above_voltage > 0.1 | Strong non-zombie: genuine channel-specific encoding |
| ΔR² > 0.1 AND R²_above_voltage > 0.05 | Moderate evidence |
| ΔR² > 0.1 AND R²_above_voltage ≈ 0 | Voltage re-encoding only — still zombie |
| ΔR² < 0.1 | Zombie at this level |

### 4.8 Causal Ablation Protocol

For every variable with ΔR² > 0.1 at any level, run the causal ablation:

```python
def causal_ablation(model, test_data, target_var, hidden_states, 
                    k_values=[0.05, 0.10, 0.20, 0.40, 0.60, 0.80],
                    n_random_repeats=10):
    """
    Progressive ablation protocol.
    
    1. Compute |r| between each hidden dim and target variable
    2. For each k (fraction of total dims):
       a. Clamp top-k% target-correlated dims to their mean
       b. Run forward pass, measure output metric
       c. Clamp k% RANDOM dims, measure output metric (10 repeats)
       d. Compute z-score = (target_delta - random_mean) / random_std
    3. Report breaking point and z-scores
    
    Output metric: cross-condition correlation (NOT per-trial temporal!)
    """
    n_dims = hidden_states.shape[-1]
    
    # Compute correlations with target
    correlations = np.array([
        np.corrcoef(hidden_states[:, d], target_var)[0, 1]
        for d in range(n_dims)
    ])
    ranked_dims = np.argsort(np.abs(correlations))[::-1]
    
    results = []
    for k_frac in k_values:
        k = int(k_frac * n_dims)
        
        # Target-clamp
        target_clamp_dims = ranked_dims[:k]
        target_cc = forward_with_clamp(model, test_data, target_clamp_dims)
        
        # Random-clamp control
        random_ccs = []
        for _ in range(n_random_repeats):
            random_dims = np.random.choice(n_dims, k, replace=False)
            random_ccs.append(forward_with_clamp(model, test_data, random_dims))
        
        z = (target_cc - np.mean(random_ccs)) / np.std(random_ccs)
        results.append({
            'k': k, 'k_frac': k_frac, 
            'target_cc': target_cc,
            'random_cc_mean': np.mean(random_ccs),
            'random_cc_std': np.std(random_ccs),
            'z_score': z,
            'verdict': 'CAUSAL' if z < -2 else 'BYPRODUCT'
        })
    
    return results
```

**Classification of mandatory variables by redundancy type:**

| Type | Breaking point | Example from hippocampus | L5PC prediction |
|------|---------------|------------------------|-----------------|
| Concentrated | Breaks at <10% clamp | g_NMDA_SC (3.9%) | Somatic I_Na, burst/tonic ratio |
| Distributed | Breaks at 40–60% clamp | gamma_amp at h=128 (39%) | BAC index, Ca hotzone activation |
| Redundant | Breaks at >70% clamp | gamma_amp at h=512 (78%) | Expected at high hidden sizes |

---

## 5. Phase 2 Implementation: Full Hay Model

### 5.1 When to Proceed to Phase 2

Proceed only if Phase 1 (Bahl) produces at least one of:
- Level B non-zombie variables (effective conductances with ΔR² > 0.2 above voltage baseline)
- Level C mandatory variables (emergent properties with high ΔR² and causal ablation z < -2)

If Phase 1 shows universal zombie at all three levels, the full Hay model will not change the conclusion and compute is better spent elsewhere.

### 5.2 Full Hay Model Recording

Follow the blueprint from Document 3. Key parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | Hay et al. 2011 (ModelDB 139653) | Gold standard L5PC |
| Compartments | ~639 | Full dendritic tree |
| Trials | 2,000 (start), scale to 10,000 | Storage-aware |
| Resolution | 1.0 ms | Matches TCN operating resolution |
| Recording vectors | ~5,000 (compartment vars) | Excluding synaptic vars |
| Storage per trial | ~11.4 MB | Strategy B' from blueprint |
| Total storage | ~23 GB (2K trials) / ~114 GB (10K trials) | Feasible |

### 5.3 Spatial Analysis

The full Hay model enables spatial questions the Bahl model cannot answer:

1. **Spatial localisation of mandatory variables**: If G_CaHVA is mandatory, is it the hot zone (685–885 μm) that matters, or the whole tree? Run Ridge ΔR² separately for hot zone compartments vs non-hot-zone compartments.

2. **Distance-dependent zombie gradient**: Plot ΔR² as a function of distance from soma. Does the network's biological encoding decrease with distance (proximal = non-zombie, distal = zombie)?

3. **Ih gradient encoding**: Ih increases exponentially with distance along the apical dendrite. Does the network discover this gradient, or just a flat average?

### 5.4 Pre-trained TCN Option

The Beniaguev pre-trained TCN weights (AUC 0.9911) are available on Kaggle in Keras/TensorFlow format. These can be loaded directly for probing without retraining:

```python
# Load pre-trained Beniaguev TCN
import tensorflow as tf
model = tf.keras.models.load_model('NMDA_TCN__DxWxT_7x128x153.h5')

# Extract hidden activations (7 layers × 128 features = 896 dims)
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
hidden_states = activation_model.predict(test_inputs)
```

**Important**: Pre-trained weights only work with the full 639-compartment Hay model input format. They do NOT work with the Bahl reduced model. Phase 1 (Bahl) requires training from scratch; Phase 2 (Hay) can leverage pre-trained weights.

For the untrained baseline, create an identical architecture with random initialisation.

---

## 6. Phase 3 Implementation: Circuit Integration

### 6.1 Minimum Viable Cortical Circuit

Following the four-level validation framework:

| Cell type | Count | Source | Role |
|-----------|-------|--------|------|
| L5 thick-tufted PC | 25–40 | Hay et al. 2011 (ModelDB 139653) | **Replacement targets** |
| PV+ basket cell | 15–20 | Dura-Bernal M1 (ModelDB 260015) | Perisomatic inhibition, gamma |
| SOM+ Martinotti | 5–10 | Dura-Bernal M1 | Dendritic inhibition |
| L2/3 pyramidal | 15–25 | Standard multicompartment | Feedforward input |
| L4 spiny stellate | 10–15 | Standard | Thalamic input relay |

**Total: ~100–150 neurons**

### 6.2 Graded Replacement Protocol

| Replacement fraction | # L5PCs replaced | Replicates |
|---------------------|------------------|------------|
| 0% (baseline) | 0 | 1 |
| 1 cell | 1 | 10 random selections |
| 10% | 3–4 | 10 |
| 25% | 6–10 | 10 |
| 50% | 12–20 | 10 |
| 100% | All | 1 |

### 6.3 Four-Level Validation Metrics

**Level 1 — Output Fidelity**
- Spike rate preservation (per cell)
- ISI distribution match (KS test)
- Victor-Purpura spike distance
- Cross-condition correlation (the metric that works)

**Level 2 — Circuit Integration**
- Gamma oscillation power (30–80 Hz PSD)
- Beta oscillation power (15–30 Hz PSD)
- Pairwise cross-correlations
- Fano factor stability

**Level 3 — Mechanistic Correspondence (DESCARTES)**
- Ridge ΔR² for Level B and C targets (from Phase 1)
- Causal ablation results
- Progressive ablation redundancy profiles

**Level 4 — Consciousness-Relevant Metrics**
- BAC firing index across replacement fractions
- PCI (Lempel-Ziv complexity of perturbation response)
- Transfer entropy (directional information flow, especially feedback)
- Approximate Φ for small L5PC subsystems (8–10 neurons via PyPhi)

### 6.4 The Key Prediction

A feedforward TCN surrogate should:
- **Pass** Level 1 (correct spikes — Beniaguev showed this)
- **Pass** Level 2 partially (gamma preserved via PING, beta may degrade)
- **Show** Level 3 mixed results (some mandatory variables preserved, some lost)
- **Fail** Level 4 selectively (Φ drops because feedforward has Φ=0 by IIT; PCI may still pass; BAC index may fail if coincidence detection is lost)

If this prediction holds, the result demonstrates that **functional equivalence (Level 1) does not guarantee computational equivalence (Level 3) or consciousness-relevant equivalence (Level 4)** — the central thesis of DESCARTES and the substrate independence question.

---

## 7. Cross-Circuit Comparison

The ultimate deliverable: a unified table across all three circuits.

```
Circuit      | Total vars | Level A   | Level B          | Level C           | Mandatory type
             |            | (gates)   | (conductances)   | (emergent)        |
-------------|-----------|-----------|------------------|-------------------|----------------
Thalamic     | 160       | 0/160     | NOT TESTED (*)   | NOT TESTED        | Unknown
Hippocampal  | 25        | 0/25(**) | N/A (no channels)| 2/~10             | gamma(holistic) + NMDA(concentrated)
L5PC (Bahl)  | ~90       | ?/~30    | ?/~25            | ?/~12             | Prediction: BAC + Ca_hotzone
L5PC (Hay)   | ~5,000    | ?/~2,500 | ?/~1,000         | ?/~12             | Spatial resolution added

(*) Thalamic effective conductances should be computed retroactively from existing data
(**) Hippocampal "gates" are actually synaptic/network variables, not HH gates —
     identifiability argument applies differently
```

### 7.1 Retroactive Thalamic Analysis

The thalamic A-R3 data (160 gating variables, R² ≈ 0) should be reanalysed with effective conductances. The Le Masson model has ~6 channel types with recorded individual gates. Compute G_eff = ḡ·mᵖ·hᵍ for each and re-probe. If R² becomes substantial, the thalamic "zombie" finding transforms into "zombie at the wrong level, non-zombie at the identifiable level."

### 7.2 The Substrate Independence Verdict

The framework resolves the zombie question into a precise hierarchy:

| Level | What's preserved | What's required | Verdict |
|-------|-----------------|-----------------|---------|
| Input-output | Same spikes | Any architecture | Functional equivalence |
| Identifiable effective states | Same conductances/currents | Architecture that discovers physics | **Mechanistic equivalence** |
| Individual gates | Same m, h, n decomposition | Impossible (non-identifiable) | **Not a valid requirement** |
| Emergent properties | Same BAC, gamma, burst/tonic | Architecture + training that discovers dynamics | **Computational equivalence** |

A substrate is adequate for prosthetic replacement if it achieves mechanistic equivalence (Level B non-zombie) and computational equivalence (Level C non-zombie). Gate-level equivalence is not required by physics. Full zombie at all levels means the substrate solves the problem through alien computation — the deepest form of the zombie problem with direct implications for consciousness preservation.

---

## 8. Project Structure

```
L5PC/
├── setup.sh                    # Vast.ai setup: NEURON, deps, model downloads
├── requirements.txt
├── l5pc/
│   ├── __init__.py
│   ├── config.py               # All hyperparameters, paths, channel specs
│   │
│   ├── simulation/             # NEURON simulation layer
│   │   ├── bahl_model.py       # Phase 1: 6-compartment Bahl L5PC
│   │   ├── hay_model.py        # Phase 2: 639-compartment Hay L5PC
│   │   ├── stimulation.py      # Synaptic input protocols (5 conditions)
│   │   ├── recording.py        # 3-level biological variable recording
│   │   └── circuit.py          # Phase 3: 100-neuron NetPyNE circuit
│   │
│   ├── surrogates/             # Neural network surrogates
│   │   ├── lstm.py             # LSTM architecture (cross-circuit comparison)
│   │   ├── tcn.py              # TCN architecture (Beniaguev replication)
│   │   ├── train.py            # Training loop with early stopping
│   │   └── extract_hidden.py   # Hidden state extraction (trained + untrained)
│   │
│   ├── probing/                # DESCARTES zombie test
│   │   ├── ridge_probe.py      # Ridge ΔR² with CV, preprocessing sweep
│   │   ├── baselines.py        # Voltage-only + temporal baselines
│   │   ├── ablation.py         # Causal + progressive ablation
│   │   └── classify.py         # Variable classification (zombie/mandatory/byproduct)
│   │
│   ├── analysis/               # Post-hoc computation
│   │   ├── effective_conductances.py  # Level B: G_eff = ḡ·mᵖ·hᵍ
│   │   ├── ionic_currents.py          # Level B: I = G·(V - E)
│   │   ├── emergent_properties.py     # Level C: BAC, burst ratio, Ca hotzone
│   │   └── cross_circuit.py           # Cross-circuit comparison table
│   │
│   ├── validation/             # Phase 3: Four-level validation
│   │   ├── level1_output.py    # Spike fidelity metrics
│   │   ├── level2_circuit.py   # Oscillation power, correlations
│   │   ├── level3_descartes.py # Ridge ΔR² + ablation
│   │   └── level4_consciousness.py  # PCI, Φ, transfer entropy
│   │
│   ├── visualization/          # Plotting
│   │   ├── probe_tables.py     # 3-level results tables
│   │   ├── ablation_curves.py  # Progressive ablation plots
│   │   ├── spatial_maps.py     # Phase 2: dendrite-overlaid ΔR²
│   │   └── replacement_curves.py  # Phase 3: graded replacement
│   │
│   └── utils/
│       ├── io.py               # HDF5/numpy save/load
│       └── metrics.py          # Cross-condition correlation, KS test, etc.
│
├── scripts/                    # CLI entry points
│   ├── run_phase1.sh           # Shell wrapper
│   ├── run_phase1.py           # Python orchestrator (skip completed steps)
│   ├── run_phase2.sh
│   ├── run_phase2.py
│   ├── run_phase3.sh
│   └── run_phase3.py
│
├── data/                       # Generated data (gitignored)
│   ├── models/                 # Downloaded NEURON models (Bahl, Hay)
│   ├── bahl_trials/            # Phase 1 simulation output
│   ├── hay_trials/             # Phase 2 simulation output
│   ├── surrogates/             # Trained model checkpoints + hidden states
│   └── results/                # Probing/ablation JSON results
│
└── docs/
    └── plans/
        ├── L5PC_ULTIMATE_GUIDE.md   # This document
        └── PROJECT_H_SELF.docx      # Future extension
```

### Design Principles

**Separation of concerns.** Each subdirectory handles one stage of the pipeline: `simulation/` generates data, `surrogates/` trains models, `probing/` tests for zombie encoding, `analysis/` computes derived quantities, `validation/` handles circuit-level metrics. Modules can be tested and run independently.

**Smart checkpointing via orchestrator scripts.** Each `run_phaseN.py` is a single Python entry point that calls simulation → training → probing → ablation in sequence, skipping any step whose output already exists. This pattern proved essential in the hippocampal experiment — the h=512 pipeline automatically skipped training when it found an existing checkpoint and jumped straight to probing. Example orchestrator logic:

```python
# scripts/run_phase1.py
def main():
    # Step 1: Simulate
    trial_dir = Path('data/bahl_trials')
    if not (trial_dir / 'trial_499.npz').exists():
        print("=== STEP 1: Simulating 500 Bahl trials ===")
        from l5pc.simulation.bahl_model import run_all_trials
        run_all_trials(n_trials=500, output_dir=trial_dir)
    else:
        print("=== STEP 1: SKIP (trials exist) ===")
    
    # Step 2: Train surrogates
    for h in [64, 128, 256]:
        ckpt = Path(f'data/surrogates/lstm_h{h}_best.pt')
        if not ckpt.exists():
            print(f"=== STEP 2: Training LSTM h={h} ===")
            from l5pc.surrogates.train import train_lstm
            train_lstm(hidden_size=h, data_dir=trial_dir, save_path=ckpt)
        else:
            print(f"=== STEP 2: SKIP h={h} (checkpoint exists) ===")
    
    # Step 3: Extract hidden states
    for h in [64, 128, 256]:
        hidden_path = Path(f'data/surrogates/lstm_h{h}_hidden.npz')
        if not hidden_path.exists():
            print(f"=== STEP 3: Extracting hidden states h={h} ===")
            from l5pc.surrogates.extract_hidden import extract
            extract(h, trained=True, save_path=hidden_path)
            extract(h, trained=False, save_path=hidden_path.with_name(f'lstm_h{h}_untrained.npz'))
    
    # Step 4: Compute Level B targets (effective conductances)
    level_b_path = Path('data/results/level_B_targets.npz')
    if not level_b_path.exists():
        print("=== STEP 4: Computing effective conductances ===")
        from l5pc.analysis.effective_conductances import compute_all
        compute_all(trial_dir, level_b_path)
    
    # Step 5: Ridge ΔR² probing (all 3 levels)
    for level in ['A', 'B', 'C']:
        for h in [64, 128, 256]:
            result_path = Path(f'data/results/ridge_level{level}_h{h}.json')
            if not result_path.exists():
                print(f"=== STEP 5: Probing Level {level}, h={h} ===")
                from l5pc.probing.ridge_probe import run_probe
                run_probe(level=level, hidden_size=h, save_path=result_path)
    
    # Step 6: Voltage-only baselines for Level B
    baseline_path = Path('data/results/voltage_baselines.json')
    if not baseline_path.exists():
        print("=== STEP 6: Voltage-only baselines ===")
        from l5pc.probing.baselines import run_voltage_baselines
        run_voltage_baselines(trial_dir, save_path=baseline_path)
    
    # Step 7: Causal ablation on any variable with ΔR² > 0.1
    # (reads results from Step 5, runs ablation on non-zombie vars)
    ablation_path = Path('data/results/causal_ablation.json')
    if not ablation_path.exists():
        print("=== STEP 7: Causal ablation ===")
        from l5pc.probing.ablation import run_all_ablations
        run_all_ablations(delta_threshold=0.1, save_path=ablation_path)
    
    # Step 8: Print cross-circuit comparison
    from l5pc.analysis.cross_circuit import print_comparison
    print_comparison()

if __name__ == '__main__':
    main()
```

**config.py centralises all parameters.** Every hyperparameter, file path, channel specification, and threshold lives in one file. Nothing is hardcoded in module code. This prevents the configuration drift that plagued the hippocampal experiment (different scripts using different thresholds for "learned" vs "structural").

```python
# l5pc/config.py
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
BAHL_TRIAL_DIR = DATA_DIR / 'bahl_trials'
HAY_TRIAL_DIR = DATA_DIR / 'hay_trials'
SURROGATE_DIR = DATA_DIR / 'surrogates'
RESULTS_DIR = DATA_DIR / 'results'

# === Simulation ===
N_TRIALS = 500
TRAIN_SPLIT = 350
VAL_SPLIT = 75
TEST_SPLIT = 75
SIM_DURATION_MS = 1000
RECORDING_DT_MS = 0.5  # Record every 0.5ms
NEURON_DT_MS = 0.025

# === Channel specifications (Bahl model) ===
CHANNEL_SPECS = {
    'NaTa_t':  {'gates': ['m', 'h'], 'exp': [3, 1], 'ion': 'na'},
    'Nap_Et2': {'gates': ['m', 'h'], 'exp': [3, 1], 'ion': 'na'},
    'K_Tst':   {'gates': ['m', 'h'], 'exp': [4, 1], 'ion': 'k'},
    'K_Pst':   {'gates': ['m', 'h'], 'exp': [2, 1], 'ion': 'k'},
    'SKv3_1':  {'gates': ['m'],      'exp': [1],    'ion': 'k'},
    'SK_E2':   {'gates': ['z'],      'exp': [1],    'ion': 'k'},
    'Im':      {'gates': ['m'],      'exp': [1],    'ion': 'k'},
    'Ih':      {'gates': ['m'],      'exp': [1],    'ion': 'hcn'},
    'Ca_HVA':  {'gates': ['m', 'h'], 'exp': [2, 1], 'ion': 'ca'},
    'Ca_LVAst':{'gates': ['m', 'h'], 'exp': [2, 1], 'ion': 'ca'},
}

# === Surrogate training ===
HIDDEN_SIZES = [64, 128, 256]
N_LAYERS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 10
LR_FACTOR = 0.5
EARLY_STOP_PATIENCE = 15
MAX_EPOCHS = 200
BATCH_SIZE = 32

# === Probing ===
RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]
CV_FOLDS = 5
PREPROCESSING_OPTIONS = ['Raw', 'StandardScaler', 'PCA_5', 'PCA_10', 'PCA_20', 'PCA_50']
SELECTIVITY_PERMS = 20
DELTA_THRESHOLD_LEARNED = 0.1  # ΔR² above this = non-zombie candidate
DELTA_THRESHOLD_STRONG = 0.2   # ΔR² above this = strong non-zombie

# === Ablation ===
ABLATION_K_FRACTIONS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
ABLATION_N_RANDOM = 10
CAUSAL_Z_THRESHOLD = -2.0  # z-score below this = causal

# === Stimulation conditions ===
STIM_CONDITIONS = {
    'subthreshold': {'basal_hz': (1, 3),  'apical_hz': (0, 1),  'n_trials': 50},
    'tonic':        {'basal_hz': (5, 15), 'apical_hz': (0, 3),  'n_trials': 100},
    'burst':        {'basal_hz': (15, 30),'apical_hz': (10, 20),'n_trials': 100},
    'bac':          {'basal_hz': (5, 10), 'apical_hz': (10, 20),'n_trials': 100},
    'mixed':        {'basal_hz': (0, 30), 'apical_hz': (0, 20), 'n_trials': 150},
}
```

---

## 9. Compute Estimates and Timeline

### Phase 1: Bahl Reduced Model

| Step | Hardware | Time | Storage |
|------|----------|------|---------|
| NEURON setup + validation | CPU | 1 hour | Minimal |
| Simulate 500 trials with full recording | CPU (8 cores) | 30 min | ~100 MB |
| Train LSTM (3 sizes) + TCN | RTX 5070 GPU | 2–3 hours | ~500 MB checkpoints |
| Extract hidden states (trained + untrained) | GPU | 30 min | ~2 GB |
| Ridge ΔR² probing (3 levels, ~70 vars) | CPU | 1–2 hours | ~50 MB results |
| Voltage-only and temporal baselines | CPU | 30 min | ~10 MB |
| Causal ablation + progressive ablation | GPU | 30 min per variable | ~10 MB |
| **Phase 1 Total** | | **~1 day** | **~3 GB** |

### Phase 2: Full Hay Model

| Step | Hardware | Time | Storage |
|------|----------|------|---------|
| Simulate 2,000 trials with full recording | CPU (8 cores parallel) | 2–4 hours | ~23 GB |
| Train LSTM + load pre-trained TCN | GPU | 3–4 hours | ~2 GB |
| Ridge ΔR² probing (spatial analysis) | CPU | 4–8 hours | ~500 MB |
| Progressive ablation sweep | GPU | 2–3 hours | ~100 MB |
| **Phase 2 Total** | | **~2 days** | **~26 GB** |

### Phase 3: Circuit Integration

| Step | Hardware | Time | Storage |
|------|----------|------|---------|
| Build 100-neuron circuit in NetPyNE | CPU | 2–3 days (tuning E/I balance) | Minimal |
| Simulate baseline (1 sec bio time) | CPU | 6–12 hours | ~5 GB |
| Graded replacement sweep (7 fractions × 10 replicates) | CPU + GPU | 5–10 days | ~50 GB |
| Four-level metric computation | CPU | 1–2 days | ~5 GB |
| **Phase 3 Total** | | **~2–3 weeks** | **~60 GB** |

### Full Project Timeline

| Period | Phase | Deliverable |
|--------|-------|-------------|
| Weeks 1–2 | Phase 1 (Bahl) | Three-level zombie test, causal ablation, cross-circuit table |
| Weeks 3–5 | Phase 2 (Hay) | Spatial analysis, pre-trained TCN probing, full dendrite map |
| Weeks 6–10 | Phase 3 (Circuit) | Graded replacement with 4-level validation |
| Weeks 11–14 | Analysis + writing | Paper integrating thalamic + hippocampal + L5PC results |

---

## 10. Dependencies

```bash
# Core simulation
pip install neuron          # NEURON simulator
pip install netpyne         # Circuit building (Phase 3)

# Machine learning
pip install torch           # Surrogate training
pip install tensorflow      # Loading pre-trained Beniaguev TCN
pip install scikit-learn    # Ridge probing

# Analysis
pip install numpy scipy matplotlib seaborn h5py

# Information theory (Phase 3, Level 4)
pip install antropy         # Lempel-Ziv for PCI
# JIDT (Java) for transfer entropy — requires Java runtime
# PyPhi for integrated information — pip install pyphi
```

---

## 11. Critical Design Notes

1. **Probe effective conductances and currents, not individual gates.** Individual gating variables are non-identifiable — R² ≈ 0 is expected, not evidence of zombie-ness. The identifiability boundary defines what counts as genuine mechanistic encoding.

2. **Separate basal and apical input channels.** BAC firing depends on coincidence detection between bottom-up (basal) and top-down (apical) input. Combining them into a single input vector prevents the surrogate from learning coincidence detection and makes Level C probing for BAC index meaningless.

3. **Use cross-condition correlation, not per-trial temporal correlation, as the output metric.** The hippocampal experiment showed per-trial temporal correlation can be near zero even when the model performs well. Cross-condition correlation captures whether the model distinguishes different input conditions.

4. **Causal ablation is mandatory, not optional.** ΔR² alone cannot distinguish learned-and-used from learned-and-ignored. The hippocampal gamma_amp initially appeared causal at h=128 but became a redundant encoding at h=512 under standard ablation. Only progressive ablation revealed it was still collectively necessary (breaks at 78% clamp).

5. **The voltage-only baseline is essential for Level B probing.** Effective conductances are voltage-dependent — a network that merely re-encodes voltage will show high R² for conductances trivially. The trained network's R² must exceed the voltage-only baseline to count as genuine channel-specific encoding.

6. **Start with Bahl, scale to Hay.** The Bahl 6-compartment model runs 100× faster and has ~90 variables (comparable to the thalamic circuit). If universal zombie at all levels with Bahl, the full Hay model won't rescue the finding.

7. **The pre-trained Beniaguev TCN is a free control.** Load it, probe it, compare against your LSTM. If the TCN (trained by Beniaguev for spike prediction) encodes effective conductances while the LSTM doesn't (or vice versa), that's an architecture-dependent zombie result — directly relevant to prosthetic design choices.

8. **BAC firing is the L5PC's gamma_amp.** It's an emergent property arising from multi-channel, multi-compartment interaction that cannot be captured by random projections. Prediction: BAC index will be mandatory (high ΔR², causal under ablation) across architectures and hidden sizes.

9. **The identifiability correction applies retroactively to the thalamic results.** Compute effective conductances from the existing A-R3 data and re-probe. This single analysis could reclassify the thalamic finding from "universal zombie" to "zombie at the wrong level."

10. **Phase 3 Level 4 (consciousness metrics) is the highest-impact finding.** If a TCN that perfectly matches spike output nevertheless reduces Φ and eliminates BAC firing, this demonstrates that functional equivalence does not guarantee phenomenal equivalence — the strongest empirical argument for the relevance of the zombie problem to consciousness science.
