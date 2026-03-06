# L5PC DESCARTES: Zombie Test for Neural Surrogates

**Do neural surrogates encode biological intermediates, or solve via alien computation?**

This repository implements the DESCARTES (Detailed Evaluation of Surrogate Computation Against Real Targets and Encoded States) framework for testing whether LSTM surrogates of layer-5 pyramidal cells (L5PCs) represent biologically meaningful intermediate variables or function as "zombies" that bypass real biophysics.

## The Zombie Test

A surrogate is a **zombie** if it achieves accurate input-output mapping without internally representing the biological variables (ion channel gates, effective conductances, dendritic calcium) that the real neuron computes. DESCARTES distinguishes four variable categories:

| Category | Meaning | Detection |
|----------|---------|-----------|
| **ZOMBIE** | Not represented beyond chance | dR2 < 0.1 |
| **VOLTAGE RE-ENCODING** | Decodable, but only via voltage | R2 does not exceed voltage baseline |
| **LEARNED BYPRODUCT** | Represented but not causally used | Passes probing but fails ablation |
| **MANDATORY** | Causally required for computation | Target-correlated clamping breaks output (z < -2) |

Only **MANDATORY** variables confirm genuine biological representation.

## Three-Phase Pipeline

### Phase 1: Bahl Reduced Model (6-compartment L5PC)
The primary validation pipeline with 8 automated steps:

```
Step 1  Simulate       500 Bahl L5PC trials (NEURON + BBP ion channels)
Step 2  Train          LSTM surrogates (h=64, 128, 256)
Step 3  Extract        Hidden states (trained + untrained baselines)
Step 4  Probe          3-level Ridge dR2 probing
Step 5  Baselines      Voltage-only controls
Step 6  Ablation       Progressive clamping (causal test)
Step 7  Classify       Final variable categorisation
Step 8  Visualise      Tables, bar charts, ablation curves
```

### Phase 2: Hay Detailed Model (639-compartment L5PC)
Scales the analysis to a biophysically detailed morphology with 2000 trials.

### Phase 3: Circuit Integration
Tests whether surrogate-replaced cells preserve network-level dynamics (gamma oscillations, burst propagation) in a cortical microcircuit.

## Three Probing Levels

| Level | Variables | Expected Result |
|-------|-----------|-----------------|
| **A** | Individual gate variables (m, h per channel per compartment) | Mostly zombie (high-dimensional, redundant) |
| **B** | Effective conductances G_eff = gbar * prod(gate^exp) | Key test: should find mandatory variables like g_NaTs2t |
| **C** | Emergent properties (spike count, burst ratio, BAC flag, Ca integral) | Some mandatory, some byproduct |

## Project Structure

```
L5PC/
|-- l5pc/                          # Core library
|   |-- config.py                  # All hyperparameters and paths
|   |-- simulation/
|   |   |-- bahl_model.py          # 6-compartment Bahl L5PC (NEURON)
|   |   |-- hay_model.py           # 639-compartment Hay L5PC
|   |   |-- recording.py           # Gate/conductance/calcium recording
|   |   |-- stimulation.py         # Synaptic input generation
|   |   |-- run_bahl_sim.py        # Trial runner with condition mixing
|   |   +-- circuit.py             # NetPyNE microcircuit (Phase 3)
|   |-- surrogates/
|   |   |-- lstm.py                # L5PC_LSTM architecture
|   |   |-- tcn.py                 # TCN baseline (Beniaguev replication)
|   |   |-- train.py               # Training loop (AdamW + ReduceLROnPlateau)
|   |   +-- extract_hidden.py      # Hidden state extraction
|   |-- probing/
|   |   |-- ridge_probe.py         # RidgeCV dR2 probing (3 levels)
|   |   |-- baselines.py           # Voltage-only R2 baselines
|   |   |-- ablation.py            # Progressive clamping (causal test)
|   |   +-- classify.py            # Final zombie/mandatory classification
|   |-- visualization/
|   |   |-- probe_tables.py        # dR2 tables and bar charts
|   |   |-- ablation_curves.py     # Progressive ablation curve plots
|   |   |-- spatial_maps.py        # Compartment-level spatial heatmaps
|   |   +-- replacement_curves.py  # Phase 3 replacement fraction plots
|   +-- utils/
|       |-- io.py                  # Trial I/O, JSON helpers
|       +-- metrics.py             # Cross-condition correlation, filters
|-- mechanisms/                    # NEURON .mod files (11 BBP ion channels)
|-- scripts/
|   |-- run_phase1.py              # Phase 1 orchestrator (Steps 1-8)
|   |-- run_phase2.py              # Phase 2 orchestrator
|   |-- run_phase3.py              # Phase 3 orchestrator
|   +-- generate_synthetic_data.py # Synthetic data for testing without NEURON
+-- requirements.txt
```

## Quick Start

### Prerequisites

- Python 3.10+
- NEURON 8.2+ (for simulation, Linux/macOS only)
- PyTorch 2.0+
- CUDA-capable GPU recommended (training + ablation)

### Installation

```bash
git clone https://github.com/CharithL/L5PC.git
cd L5PC
pip install -r requirements.txt
```

### Running the Full Pipeline (Linux / Vast.ai)

```bash
# Mechanisms are auto-compiled in Step 1, but you can verify manually:
cd mechanisms && nrnivmodl . && cd ..

# Run all 8 steps
python scripts/run_phase1.py

# Resume from a specific step (e.g., after fixing a bug)
python scripts/run_phase1.py --start-step 5

# Re-run everything from scratch
python scripts/run_phase1.py --force
```

### Running Without NEURON (Windows / Testing)

```bash
# Generate synthetic trial data (no NEURON required)
python scripts/generate_synthetic_data.py

# Run Steps 2-8 on synthetic data
python scripts/run_phase1.py --start-step 2
```

## Configuration

All hyperparameters are centralised in `l5pc/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_TRIALS` | 500 | Total simulation trials |
| `TRAIN_SPLIT` / `VAL_SPLIT` / `TEST_SPLIT` | 350 / 75 / 75 | Data splits |
| `T_STEPS` | 2000 | Timesteps per trial (1s at 0.5ms resolution) |
| `HIDDEN_SIZES` | [64, 128, 256] | LSTM hidden dimensions |
| `DELTA_THRESHOLD_LEARNED` | 0.1 | dR2 threshold for non-zombie |
| `CAUSAL_Z_THRESHOLD` | -2.0 | z-score threshold for causal ablation |
| `ABLATION_K_FRACTIONS` | [0.05, 0.10, 0.20, 0.40, 0.60, 0.80] | Progressive clamping fractions |

## Key Methodology

### Ridge dR2 Probing
For each biophysical variable, fit RidgeCV from **trained** hidden states and from **untrained** (random-init) hidden states:

```
dR2 = R2_trained - R2_untrained
```

Variables with dR2 < 0.1 are classified as **ZOMBIE**: the network carries no more information about them than random projections would.

### Progressive Clamping Ablation
For variables that pass the dR2 threshold:

1. Rank hidden dimensions by |correlation| with the target variable
2. For increasing k (fraction of dims), clamp the top-k% target-correlated dims to their mean
3. Run forward pass, measure cross-condition correlation of output
4. Compare against random clamping (repeated N times)
5. Compute z-score: if z < -2 at any k, the variable is **MANDATORY**

The **breaking point** k classifies redundancy type:
- **Concentrated** (k < 10%): Information packed in few dimensions
- **Distributed** (10-60%): Spread across many dimensions
- **Redundant** (k > 60%): Highly redundant encoding

## Ion Channel Mechanisms

The 11 BBP ion channel `.mod` files in `mechanisms/`:

| Channel | Ion | Compartments | Role |
|---------|-----|-------------|------|
| NaTa_t | Na+ | soma, trunk, nexus, tuft | Fast transient sodium (AP initiation) |
| Nap_Et2 | Na+ | soma, trunk | Persistent sodium |
| K_Pst | K+ | soma | Persistent potassium |
| K_Tst | K+ | soma | Transient potassium |
| SKv3_1 | K+ | soma, trunk, nexus, tuft | Fast delayed rectifier |
| SK_E2 | K+ | soma, nexus | Ca-activated potassium |
| Im | K+ | soma, trunk, nexus | Muscarinic potassium |
| Ih | HCN | all | Hyperpolarisation-activated cation |
| Ca_HVA | Ca2+ | soma, nexus | High-voltage-activated calcium |
| Ca_LVAst | Ca2+ | nexus, tuft | Low-voltage-activated calcium (BAC firing) |
| CaDynamics_E2 | -- | soma, nexus | Calcium dynamics (decay/buffering) |

## Stimulation Conditions

Trials span five regimes to probe the full operating range:

| Condition | Basal Hz | Apical Hz | Trials | Expected Behavior |
|-----------|----------|-----------|--------|-------------------|
| Subthreshold | 1-3 | 0-1 | 50 | No spikes (passive response) |
| Tonic | 5-15 | 0-3 | 100 | Regular spiking |
| Burst | 15-30 | 10-20 | 100 | Burst firing |
| BAC | 5-10 | 10-20 | 100 | Back-propagating AP-activated Ca spike |
| Mixed | 0-30 | 0-20 | 150 | Full dynamic range |

## Expected Results (Phase 1)

Based on the DESCARTES framework predictions for a 6-compartment Bahl model:

### Level A (Gates)
- Most individual gate variables (m, h) expected to be **ZOMBIE**
- High-dimensional and redundant; surrogates bypass them

### Level B (Effective Conductances) -- The Key Table
- **g_NaTa_t (soma)**: Expected **MANDATORY CONCENTRATED** -- essential for spike initiation, encoded in few dimensions
- **g_Ih**: Expected **MANDATORY DISTRIBUTED** -- critical for dendritic integration
- **g_Ca_LVAst (nexus)**: Expected **MANDATORY** -- required for BAC firing
- Some conductances may be **VOLTAGE RE-ENCODING** (decodable only through voltage correlation)

### Level C (Emergent Properties)
- **BAC firing flag**: Expected **MANDATORY** -- key L5PC computation
- **Burst ratio**: Likely **MANDATORY** or **LEARNED BYPRODUCT**
- **Spike count**: May be a **LEARNED BYPRODUCT** (represented but not causally used)

## Outputs

After a successful run, results are saved to `data/results/`:

```
data/results/
|-- ridge_level{A,B,C}_h{64,128,256}.json   # Ridge dR2 per variable
|-- voltage_baselines.json                    # Voltage-only R2 controls
|-- ablation_results.json                     # Progressive clamping results
|-- classification_summary.json               # Final variable categories
+-- figures/
    |-- delta_r2_level{A,B,C}_h{size}.png    # dR2 bar charts
    |-- cross_hidden_level{A,B,C}.png        # Cross-hidden-size comparison
    |-- ablation_{var}_h{size}.png           # Per-variable ablation curves
    +-- mandatory_summary.png                # Summary of all mandatory variables
```

## Hardware Requirements

| Step | Resource | Time Estimate |
|------|----------|---------------|
| Step 1 (Simulate) | CPU, NEURON | ~2 min (500 trials) |
| Step 2 (Train) | GPU recommended | ~10-30 min (3 models) |
| Step 3 (Extract) | GPU | ~2 min |
| Steps 4-5 (Probe) | CPU | ~5 min |
| Step 6 (Ablation) | GPU strongly recommended | ~30-120 min |
| Steps 7-8 (Classify + Visualise) | CPU | < 1 min |

Tested on: Vast.ai A10 GPU, RTX 5070/5080 (Windows, Steps 2-8 only).

## References

- Bahl et al. (2012). Automated optimization of a reduced layer 5 pyramidal cell model. *J. Neurosci. Methods*.
- Hay et al. (2011). Models of neocortical layer 5b pyramidal cells. *PLoS Comput. Biol.*
- Beniaguev et al. (2021). Single cortical neurons as deep artificial neural networks. *Neuron*.
