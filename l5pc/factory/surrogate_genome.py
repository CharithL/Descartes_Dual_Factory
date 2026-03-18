"""Surrogate genome specification and composer for the DESCARTES factory."""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SurrogateGenome_v3:
    """Complete surrogate model specification for the enhanced factory."""

    # ── Identity ──────────────────────────────────────────────────────
    genome_id: str = ''
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0

    # ── Architecture ──────────────────────────────────────────────────
    architecture: str = 'lstm'  # 'lstm', 'ude', 'ltc', 'rmm', 'mamba', 'cde', 'koopman', 'volterra'
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.0

    # ── Training ──────────────────────────────────────────────────────
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 500
    early_stopping_patience: int = 20
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd'
    scheduler: str = 'cosine'  # 'cosine', 'plateau', 'none'
    gradient_clip: float = 1.0

    # ── Loss ──────────────────────────────────────────────────────────
    primary_loss: str = 'mse'  # 'mse', 'mae', 'huber'
    auxiliary_bio_loss: str = 'none'  # 'none', 'fi_curve', 'impedance', 'spike_timing'
    aux_bio_weight: float = 0.0
    aux_bio_variables: List[str] = field(default_factory=list)

    # ── Regularization ────────────────────────────────────────────────
    weight_decay: float = 1e-4
    hidden_l1: float = 0.0
    information_bottleneck: bool = False
    ib_beta: float = 0.001
    slow_feature_penalty: float = 0.0
    disentanglement_penalty: float = 0.0

    # ── Inductive biases ──────────────────────────────────────────────
    continuous_time: bool = False
    compartmental_structure: bool = False
    excitatory_inhibitory: bool = False
    positivity_constraints: bool = False

    # ── Curriculum ────────────────────────────────────────────────────
    curriculum: str = 'none'  # 'none', 'step_current_first', 'short_to_long'
    curriculum_stages: List[dict] = field(default_factory=list)

    # ── Architecture-specific: UDE ────────────────────────────────────
    ude_known_rhs: str = 'hodgkin_huxley'
    ude_neural_residual_dim: int = 32
    ude_solver: str = 'dopri5'
    ude_adjoint: bool = True

    # ── Architecture-specific: LTC ────────────────────────────────────
    ltc_time_constant_range: Tuple[float, float] = (0.1, 10.0)
    ltc_backbone: str = 'ncps'  # 'ncps', 'cfc'

    # ── Architecture-specific: RMM ────────────────────────────────────
    rmm_memory_slots: int = 8
    rmm_read_heads: int = 2
    rmm_write_heads: int = 1

    # ── Architecture-specific: Mamba ──────────────────────────────────
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # ── Architecture-specific: CDE ────────────────────────────────────
    cde_interpolation: str = 'cubic'  # 'cubic', 'linear'
    cde_vector_field_dim: int = 64

    # ── Architecture-specific: Koopman ────────────────────────────────
    koopman_observable_dim: int = 64
    koopman_rank: int = 16

    # ── Architecture-specific: Volterra ───────────────────────────────
    volterra_order: int = 2
    volterra_memory_length: int = 50
    volterra_rank: int = 8

    def to_dict(self) -> Dict:
        """Serialize genome to dictionary."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, tuple):
                d[k] = list(v)
            else:
                d[k] = v
        return d

    def fingerprint(self) -> str:
        """Configuration-dependent unique signature."""
        return str(hash((self.architecture, self.hidden_dim, self.n_layers,
                        self.primary_loss, self.auxiliary_bio_loss,
                        self.weight_decay, self.information_bottleneck,
                        self.continuous_time, self.compartmental_structure)))


class SurrogateGenomeComposer:
    """Composes, mutates, and crosses over SurrogateGenome_v3 instances."""

    ARCHITECTURES = ['lstm', 'ude', 'ltc', 'rmm', 'mamba', 'cde', 'koopman', 'volterra']
    LOSSES = ['mse', 'mae', 'huber']
    REGULARIZERS = ['none', 'l1', 'l2', 'information_bottleneck']
    HIDDEN_DIMS = [32, 64, 128, 256, 512]
    BIO_LOSS_WEIGHTS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

    @staticmethod
    def compose_random(rng: np.random.Generator) -> SurrogateGenome_v3:
        """Compose a random genome from the search space."""
        g = SurrogateGenome_v3()

        g.architecture = rng.choice(SurrogateGenomeComposer.ARCHITECTURES)
        g.hidden_dim = int(rng.choice(SurrogateGenomeComposer.HIDDEN_DIMS))
        g.n_layers = int(rng.integers(1, 5))
        g.dropout = float(rng.choice([0.0, 0.1, 0.2, 0.3, 0.5]))

        g.learning_rate = float(10 ** rng.uniform(-4, -2))
        g.batch_size = int(rng.choice([16, 32, 64, 128]))
        g.max_epochs = int(rng.choice([200, 500, 1000]))
        g.early_stopping_patience = int(rng.choice([10, 20, 50]))
        g.optimizer = rng.choice(['adam', 'adamw', 'sgd'])
        g.scheduler = rng.choice(['cosine', 'plateau', 'none'])
        g.gradient_clip = float(rng.choice([0.5, 1.0, 5.0, 10.0]))

        g.primary_loss = rng.choice(SurrogateGenomeComposer.LOSSES)
        g.auxiliary_bio_loss = rng.choice(['none', 'fi_curve', 'impedance', 'spike_timing'])
        g.aux_bio_weight = float(rng.choice(SurrogateGenomeComposer.BIO_LOSS_WEIGHTS))

        g.weight_decay = float(10 ** rng.uniform(-6, -2))
        g.hidden_l1 = float(rng.choice([0.0, 1e-5, 1e-4, 1e-3]))
        g.information_bottleneck = bool(rng.choice([True, False]))
        g.ib_beta = float(10 ** rng.uniform(-4, -1))
        g.slow_feature_penalty = float(rng.choice([0.0, 0.001, 0.01, 0.1]))
        g.disentanglement_penalty = float(rng.choice([0.0, 0.001, 0.01, 0.1]))

        g.continuous_time = bool(rng.choice([True, False]))
        g.compartmental_structure = bool(rng.choice([True, False]))
        g.excitatory_inhibitory = bool(rng.choice([True, False]))
        g.positivity_constraints = bool(rng.choice([True, False]))

        g.curriculum = rng.choice(['none', 'step_current_first', 'short_to_long'])

        # Architecture-specific randomisation
        g.ude_neural_residual_dim = int(rng.choice([16, 32, 64]))
        g.ude_solver = rng.choice(['dopri5', 'euler', 'rk4'])
        g.ltc_time_constant_range = (float(rng.uniform(0.01, 1.0)),
                                     float(rng.uniform(1.0, 100.0)))
        g.ltc_backbone = rng.choice(['ncps', 'cfc'])
        g.rmm_memory_slots = int(rng.choice([4, 8, 16, 32]))
        g.rmm_read_heads = int(rng.choice([1, 2, 4]))
        g.rmm_write_heads = int(rng.choice([1, 2]))
        g.mamba_d_state = int(rng.choice([8, 16, 32]))
        g.mamba_d_conv = int(rng.choice([2, 4, 8]))
        g.mamba_expand = int(rng.choice([1, 2, 4]))
        g.cde_vector_field_dim = int(rng.choice([32, 64, 128]))
        g.koopman_observable_dim = int(rng.choice([32, 64, 128]))
        g.koopman_rank = int(rng.choice([8, 16, 32]))
        g.volterra_order = int(rng.choice([2, 3]))
        g.volterra_memory_length = int(rng.choice([20, 50, 100]))
        g.volterra_rank = int(rng.choice([4, 8, 16]))

        g.genome_id = g.fingerprint()
        return g

    @staticmethod
    def mutate(parent: SurrogateGenome_v3,
               mutation_rate: float = 0.3,
               rng: np.random.Generator = None) -> SurrogateGenome_v3:
        """Point-mutate a parent genome."""
        if rng is None:
            rng = np.random.default_rng()

        child = copy.deepcopy(parent)
        child.parent_ids = [parent.genome_id]
        child.generation = parent.generation + 1

        # Architecture mutation
        if rng.random() < mutation_rate:
            child.architecture = rng.choice(SurrogateGenomeComposer.ARCHITECTURES)

        # Hidden dim mutation
        if rng.random() < mutation_rate:
            child.hidden_dim = int(rng.choice(SurrogateGenomeComposer.HIDDEN_DIMS))

        # Layers mutation
        if rng.random() < mutation_rate:
            child.n_layers = int(rng.integers(1, 5))

        # Dropout mutation
        if rng.random() < mutation_rate:
            child.dropout = float(rng.choice([0.0, 0.1, 0.2, 0.3, 0.5]))

        # Learning rate mutation
        if rng.random() < mutation_rate:
            child.learning_rate = float(10 ** rng.uniform(-4, -2))

        # Batch size mutation
        if rng.random() < mutation_rate:
            child.batch_size = int(rng.choice([16, 32, 64, 128]))

        # Optimizer mutation
        if rng.random() < mutation_rate:
            child.optimizer = rng.choice(['adam', 'adamw', 'sgd'])

        # Scheduler mutation
        if rng.random() < mutation_rate:
            child.scheduler = rng.choice(['cosine', 'plateau', 'none'])

        # Loss mutation
        if rng.random() < mutation_rate:
            child.primary_loss = rng.choice(SurrogateGenomeComposer.LOSSES)

        # Auxiliary bio loss mutation
        if rng.random() < mutation_rate:
            child.auxiliary_bio_loss = rng.choice(['none', 'fi_curve', 'impedance', 'spike_timing'])
            child.aux_bio_weight = float(rng.choice(SurrogateGenomeComposer.BIO_LOSS_WEIGHTS))

        # Regularization mutations
        if rng.random() < mutation_rate:
            child.weight_decay = float(10 ** rng.uniform(-6, -2))

        if rng.random() < mutation_rate:
            child.hidden_l1 = float(rng.choice([0.0, 1e-5, 1e-4, 1e-3]))

        if rng.random() < mutation_rate:
            child.information_bottleneck = not child.information_bottleneck

        if rng.random() < mutation_rate:
            child.slow_feature_penalty = float(rng.choice([0.0, 0.001, 0.01, 0.1]))

        if rng.random() < mutation_rate:
            child.disentanglement_penalty = float(rng.choice([0.0, 0.001, 0.01, 0.1]))

        # Inductive bias mutations
        if rng.random() < mutation_rate:
            child.continuous_time = not child.continuous_time

        if rng.random() < mutation_rate:
            child.compartmental_structure = not child.compartmental_structure

        if rng.random() < mutation_rate:
            child.excitatory_inhibitory = not child.excitatory_inhibitory

        if rng.random() < mutation_rate:
            child.positivity_constraints = not child.positivity_constraints

        # Curriculum mutation
        if rng.random() < mutation_rate:
            child.curriculum = rng.choice(['none', 'step_current_first', 'short_to_long'])

        # Gradient clip mutation
        if rng.random() < mutation_rate:
            child.gradient_clip = float(rng.choice([0.5, 1.0, 5.0, 10.0]))

        child.genome_id = child.fingerprint()
        return child

    @staticmethod
    def crossover(parent_a: SurrogateGenome_v3,
                  parent_b: SurrogateGenome_v3,
                  rng: np.random.Generator = None) -> SurrogateGenome_v3:
        """Uniform crossover between two parent genomes."""
        if rng is None:
            rng = np.random.default_rng()

        child = copy.deepcopy(parent_a)
        child.parent_ids = [parent_a.genome_id, parent_b.genome_id]
        child.generation = max(parent_a.generation, parent_b.generation) + 1

        # For each field, randomly pick from parent_a or parent_b
        b = parent_b

        if rng.random() < 0.5:
            child.architecture = b.architecture
        if rng.random() < 0.5:
            child.hidden_dim = b.hidden_dim
        if rng.random() < 0.5:
            child.n_layers = b.n_layers
        if rng.random() < 0.5:
            child.dropout = b.dropout

        if rng.random() < 0.5:
            child.learning_rate = b.learning_rate
        if rng.random() < 0.5:
            child.batch_size = b.batch_size
        if rng.random() < 0.5:
            child.max_epochs = b.max_epochs
        if rng.random() < 0.5:
            child.early_stopping_patience = b.early_stopping_patience
        if rng.random() < 0.5:
            child.optimizer = b.optimizer
        if rng.random() < 0.5:
            child.scheduler = b.scheduler
        if rng.random() < 0.5:
            child.gradient_clip = b.gradient_clip

        if rng.random() < 0.5:
            child.primary_loss = b.primary_loss
        if rng.random() < 0.5:
            child.auxiliary_bio_loss = b.auxiliary_bio_loss
            child.aux_bio_weight = b.aux_bio_weight
            child.aux_bio_variables = copy.deepcopy(b.aux_bio_variables)

        if rng.random() < 0.5:
            child.weight_decay = b.weight_decay
        if rng.random() < 0.5:
            child.hidden_l1 = b.hidden_l1
        if rng.random() < 0.5:
            child.information_bottleneck = b.information_bottleneck
            child.ib_beta = b.ib_beta
        if rng.random() < 0.5:
            child.slow_feature_penalty = b.slow_feature_penalty
        if rng.random() < 0.5:
            child.disentanglement_penalty = b.disentanglement_penalty

        if rng.random() < 0.5:
            child.continuous_time = b.continuous_time
        if rng.random() < 0.5:
            child.compartmental_structure = b.compartmental_structure
        if rng.random() < 0.5:
            child.excitatory_inhibitory = b.excitatory_inhibitory
        if rng.random() < 0.5:
            child.positivity_constraints = b.positivity_constraints

        if rng.random() < 0.5:
            child.curriculum = b.curriculum
            child.curriculum_stages = copy.deepcopy(b.curriculum_stages)

        # Architecture-specific: inherit block from one parent
        if rng.random() < 0.5:
            child.ude_known_rhs = b.ude_known_rhs
            child.ude_neural_residual_dim = b.ude_neural_residual_dim
            child.ude_solver = b.ude_solver
            child.ude_adjoint = b.ude_adjoint

        if rng.random() < 0.5:
            child.ltc_time_constant_range = b.ltc_time_constant_range
            child.ltc_backbone = b.ltc_backbone

        if rng.random() < 0.5:
            child.rmm_memory_slots = b.rmm_memory_slots
            child.rmm_read_heads = b.rmm_read_heads
            child.rmm_write_heads = b.rmm_write_heads

        if rng.random() < 0.5:
            child.mamba_d_state = b.mamba_d_state
            child.mamba_d_conv = b.mamba_d_conv
            child.mamba_expand = b.mamba_expand

        if rng.random() < 0.5:
            child.cde_interpolation = b.cde_interpolation
            child.cde_vector_field_dim = b.cde_vector_field_dim

        if rng.random() < 0.5:
            child.koopman_observable_dim = b.koopman_observable_dim
            child.koopman_rank = b.koopman_rank

        if rng.random() < 0.5:
            child.volterra_order = b.volterra_order
            child.volterra_memory_length = b.volterra_memory_length
            child.volterra_rank = b.volterra_rank

        child.genome_id = child.fingerprint()
        return child
