"""
L5PC DESCARTES — Centralised Configuration

All hyperparameters, paths, channel specifications, and thresholds.
Nothing is hardcoded in module code. Change settings here only.
"""
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
BAHL_MODEL_DIR = DATA_DIR / 'models' / 'bahl'
HAY_MODEL_DIR = DATA_DIR / 'models' / 'hay'
BAHL_TRIAL_DIR = DATA_DIR / 'bahl_trials'
HAY_TRIAL_DIR = DATA_DIR / 'hay_trials'
SURROGATE_DIR = DATA_DIR / 'surrogates'
RESULTS_DIR = DATA_DIR / 'results'
TCN_PRETRAINED_PATH = DATA_DIR / 'models' / 'beniaguev_tcn' / 'NMDA_TCN__DxWxT_7x128x153.h5'

# === Simulation ===
N_TRIALS = 500
TRAIN_SPLIT = 350
VAL_SPLIT = 75
TEST_SPLIT = 75
SIM_DURATION_MS = 1000
RECORDING_DT_MS = 0.5       # Record every 0.5ms
NEURON_DT_MS = 0.025         # NEURON internal timestep
DOWNSAMPLE_FACTOR = int(RECORDING_DT_MS / NEURON_DT_MS)  # = 20
T_STEPS = int(SIM_DURATION_MS / RECORDING_DT_MS)  # = 2000

# === Synapse counts (Bahl reduced model) ===
N_BASAL_SYN = 20
N_APICAL_SYN = 20
N_SOMA_SYN = 10       # Inhibitory (perisomatic)
TOTAL_SYN = N_BASAL_SYN + N_APICAL_SYN + N_SOMA_SYN

# === Channel specifications ===
# Format: {channel_name: {gates, exponents, ion type}}
# Used for Level B effective conductance computation
CHANNEL_SPECS = {
    'NaTa_t':   {'gates': ['m', 'h'], 'exp': [3, 1], 'ion': 'na'},
    'Nap_Et2':  {'gates': ['m', 'h'], 'exp': [3, 1], 'ion': 'na'},
    'K_Tst':    {'gates': ['m', 'h'], 'exp': [4, 1], 'ion': 'k'},
    'K_Pst':    {'gates': ['m', 'h'], 'exp': [2, 1], 'ion': 'k'},
    'SKv3_1':   {'gates': ['m'],      'exp': [1],    'ion': 'k'},
    'SK_E2':    {'gates': ['z'],      'exp': [1],    'ion': 'k'},
    'Im':       {'gates': ['m'],      'exp': [1],    'ion': 'k'},
    'Ih':       {'gates': ['m'],      'exp': [1],    'ion': 'hcn'},
    'Ca_HVA':   {'gates': ['m', 'h'], 'exp': [2, 1], 'ion': 'ca'},
    'Ca_LVAst': {'gates': ['m', 'h'], 'exp': [2, 1], 'ion': 'ca'},
}

# Reversal potentials (mV) — standard values, overridden by NEURON if different
REVERSAL_POTENTIALS = {
    'na': 50.0,
    'k': -85.0,
    'ca': 132.458,  # Nernst at 2mM external, ~50nM internal, 37°C
    'hcn': -45.0,   # HCN mixed cation
}

# Bahl model compartment regions
BAHL_REGIONS = ['soma', 'basal', 'apical_trunk', 'nexus', 'tuft']

# Calcium hot zone (Hay model, distance from soma in μm)
CA_HOTZONE_START_UM = 685
CA_HOTZONE_END_UM = 885

# === Surrogate training ===
HIDDEN_SIZES = [64, 128, 256]
N_LSTM_LAYERS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 10
LR_FACTOR = 0.5
EARLY_STOP_PATIENCE = 15
MAX_EPOCHS = 200
BATCH_SIZE = 32

# TCN architecture (Beniaguev replication)
TCN_N_LAYERS = 7
TCN_N_FEATURES = 128
TCN_RECEPTIVE_FIELD = 153  # ms

# === Probing ===
RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
CV_FOLDS = 5
PREPROCESSING_OPTIONS = ['Raw', 'StandardScaler', 'PCA_5', 'PCA_10', 'PCA_20', 'PCA_50']
SELECTIVITY_PERMS = 20
P_THRESHOLD = 0.05
DELTA_THRESHOLD_LEARNED = 0.1    # ΔR² above this = non-zombie candidate
DELTA_THRESHOLD_STRONG = 0.2     # ΔR² above this = strong non-zombie
VOLTAGE_ABOVE_THRESHOLD = 0.1    # R² above voltage baseline for genuine encoding
VOLTAGE_ABOVE_MODERATE = 0.05

# === Ablation ===
ABLATION_K_FRACTIONS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
ABLATION_N_RANDOM = 10
CAUSAL_Z_THRESHOLD = -2.0  # z-score below this = causal

# === Stimulation conditions ===
STIM_CONDITIONS = {
    'subthreshold': {'basal_hz': (1, 3),   'apical_hz': (0, 1),   'n_trials': 50},
    'tonic':        {'basal_hz': (5, 15),  'apical_hz': (0, 3),   'n_trials': 100},
    'burst':        {'basal_hz': (15, 30), 'apical_hz': (10, 20), 'n_trials': 100},
    'bac':          {'basal_hz': (5, 10),  'apical_hz': (10, 20), 'n_trials': 100},
    'mixed':        {'basal_hz': (0, 30),  'apical_hz': (0, 20),  'n_trials': 150},
}

# === Phase 2: Hay model ===
HAY_N_TRIALS = 2000
HAY_RECORDING_DT_MS = 1.0  # Coarser for storage
HAY_N_COMPARTMENTS = 639

# === Phase 3: Circuit integration ===
CIRCUIT_CELL_COUNTS = {
    'L5PC':       35,     # Replacement targets
    'PV_basket':  18,     # Perisomatic inhibition
    'SOM_martinotti': 8,  # Dendritic inhibition
    'L23_pyr':    20,     # Feedforward input
    'L4_stellate': 12,    # Thalamic relay
}
REPLACEMENT_FRACTIONS = [0.0, 'single', 0.10, 0.25, 0.50, 1.0]
REPLACEMENT_REPLICATES = 10

# Oscillation bands (Hz)
GAMMA_BAND = (30, 80)
BETA_BAND = (15, 30)

# Burst detection
BURST_ISI_THRESHOLD_MS = 10.0

# === SAE (Section 3) ===
SAE_EXPANSION_FACTORS = [4, 8, 16]
SAE_K = 20
SAE_LR = 3e-4
SAE_EPOCHS = 50
SAE_BATCH_SIZE = 4096
SAE_DEAD_FEATURE_THRESHOLD = 0.80

# === MLP Probe (Section 4) ===
MLP_PROBE_HIDDEN_DIM = 64
MLP_PROBE_EPOCHS = 50
MLP_PROBE_LR = 1e-3
MLP_NONLINEAR_GAIN_THRESHOLD = 0.1

# === Statistical Hardening (Section 5) ===
HARDENING_N_BLOCK_PERMS = 500
HARDENING_N_PHASE_SURROGATES = 200
HARDENING_N_CIRCULAR_SHIFTS = 1000
HARDENING_MIN_BLOCK_SIZE = 50
HARDENING_IAAFT_ITERATIONS = 20
HARDENING_FDR_ALPHA = 0.05
HARDENING_TOST_EQUIVALENCE_BOUND = 0.05
HARDENING_BF_PRIOR_SCALE = 0.1
HARDENING_GAP_SIZE = 50

# === Frequency Bands (Section 5.6) ===
FREQUENCY_BANDS = {
    'ultra_slow': (0.1, 1),
    'slow': (1, 10),
    'medium': (10, 100),
    'fast': (100, 450),
}

# === Probe Tiers ===
CCA_N_COMPONENTS = 10
CCA_N_PERMUTATIONS = 200
RSA_N_SAMPLES = 2000
KOOPMAN_N_MODES = 20
DSA_DELAY = 1
DSA_RANK = 20
TDA_MAX_DIM = 2
TDA_N_LANDMARKS = 200
MINE_EPOCHS = 200
MINE_BATCH_SIZE = 512
MDL_N_PORTIONS = 10
TEMPORAL_WINDOW_SIZES = [1, 5, 10, 20, 50]

# === Derived ===
def get_total_input_dim():
    return N_BASAL_SYN + N_APICAL_SYN + N_SOMA_SYN
