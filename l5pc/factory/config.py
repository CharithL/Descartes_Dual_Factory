"""Factory-specific constants for the DESCARTES Dual Factory v3.0."""

# Thompson sampling priors (Beta distribution)
THOMPSON_ALPHA_PRIOR = 1  # Initial success count
THOMPSON_BETA_PRIOR = 1   # Initial failure count

# DreamCoder schedule
DREAMCODER_WAKE_INTERVAL = 25  # Run wake phase every N rounds
DREAMCODER_MIN_PATTERN_OCCURRENCES = 5  # Minimum samples before pruning
DREAMCODER_PRUNE_THRESHOLD = 0.2  # Remove patterns below this fitness

# LLM balloon
LLM_BALLOON_THRESHOLD = 50  # Stale rounds before triggering LLM
LLM_MAX_TOKENS = 2000

# LLM backend: 'anthropic' or 'ollama'
LLM_BACKEND = 'ollama'  # Default to local Ollama
LLM_MODEL_ANTHROPIC = 'claude-sonnet-4-20250514'
LLM_MODEL_OLLAMA = 'llama3.1'  # Change to your preferred Ollama model
OLLAMA_BASE_URL = 'http://localhost:11434'

# Output validation gate
OUTPUT_CC_THRESHOLD = 0.7  # Minimum cross-condition correlation to proceed

# Fitness weights (alpha + beta + gamma = 1.0)
FITNESS_ALPHA = 0.3   # Output accuracy
FITNESS_BETA = 0.5    # Biological correspondence
FITNESS_GAMMA = 0.2   # Causal necessity

# Campaign phase boundaries (as fraction of total rounds)
PHASE_1_END = 0.25    # Template + random
PHASE_2_END = 0.50    # Mutation + crossover
PHASE_3_END = 0.75    # DreamCoder library
# Phase 4: LLM balloon (0.75-1.0)

# Architecture options
ARCHITECTURES = [
    'lstm', 'gru', 'neural_ode', 'ude', 'ltc', 'rmm',
    'mamba', 'neural_cde', 'koopman_ae', 'volterra',
    'tcn', 'transformer', 'pinn'
]

# Hidden dim options
HIDDEN_DIMS = [16, 32, 64, 128, 256, 512]

# Bio loss weight options
BIO_LOSS_WEIGHTS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

# Loss functions
LOSSES = ['mse', 'bce', 'poisson_nll', 'huber']

# Regularizer options
REGULARIZERS = {
    'weight_decay': [0.0, 1e-5, 1e-4, 1e-3],
    'hidden_l1': [0.0, 1e-4, 1e-3, 1e-2],
    'slow_feature_penalty': [0.0, 0.01, 0.1, 1.0],
    'disentanglement_penalty': [0.0, 0.01, 0.1],
}
