"""Probe genome specification for the DESCARTES probing factory."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProbeGenome_v3:
    """Complete probe specification for the enhanced factory."""

    # Identity
    genome_id: str = ''
    tier: int = 0
    probe_type: str = 'ridge'

    # Tier 0 transform (applied before probe)
    transform: str = 'raw'  # 'raw', 'sae', 'koopman', 'pca', 'zscore'
    transform_params: dict = field(default_factory=dict)

    # Probe configuration
    decoder_type: str = 'ridge'  # 'ridge', 'lasso', 'mlp_1', 'mlp_2', 'knn'
    decoder_params: dict = field(default_factory=dict)

    # Target specification
    target_type: str = 'single'  # 'single', 'joint', 'conditional'
    target_names: List[str] = field(default_factory=list)
    condition: Optional[str] = None

    # Temporal specification
    temporal_mode: str = 'snapshot'  # 'snapshot', 'window', 'trajectory'
    window_size: int = 1

    # Statistical hardening
    null_method: str = 'block_permutation'
    n_permutations: int = 1000
    fdr_correction: bool = True

    # Gate-specific (LSTM only)
    gate_target: Optional[str] = None  # 'forget', 'input', 'output', 'cell'

    def fingerprint(self) -> str:
        """Architecture/neuron-independent probe signature."""
        return str(hash((self.probe_type, self.transform,
                        self.decoder_type, self.temporal_mode,
                        self.null_method)))
