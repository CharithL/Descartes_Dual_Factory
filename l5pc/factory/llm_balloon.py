"""LLM Balloon Expander for the DESCARTES surrogate factory.

Uses an LLM to propose novel surrogate architectures and probing strategies,
expanding the search space beyond what combinatorial enumeration can reach.
"""

import json
import logging
from typing import List, Optional

from .surrogate_genome import SurrogateGenome_v3, SurrogateGenomeComposer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt constants
# ---------------------------------------------------------------------------

SYSTEM_SURROGATE_BALLOON = """\
You are a computational neuroscience architect designing surrogate neural \
network models for a Layer 5 Pyramidal Cell (L5PC) biophysical simulation.

Circuit context:
{circuit_context}

Architectures already tested:
{architectures_tested}

Current best biophysical fidelity score: {best_bio_score}
Current best output cross-correlation: {best_output_cc}
Number of genomes evaluated so far: {n_genomes}
Verdict distribution across tested genomes: {verdict_distribution}

Your task: propose 3-5 genuinely novel architecture configurations that have \
NOT yet been tried. Think beyond standard RNN/LSTM/GRU. Consider:

- Structured state-space models (S4, Mamba) that handle long-range temporal \
  dependencies without the Volterra vs LSTM zombie risk (memorising dynamics \
  without encoding mechanism).
- Hamiltonian Neural Networks that embed energy-conservation priors matching \
  ion-channel biophysics.
- Neural Operators (DeepONet, Fourier Neural Operator) that learn the \
  solution operator of the cable equation directly.
- Mechanistic Neural Networks that blend learned components with known \
  Hodgkin-Huxley gate kinetics.
- Curriculum learning schedules that progress from passive to active regimes.
- Knowledge distillation from an ensemble of existing surrogates.
- Causal discovery layers that enforce directed information flow matching \
  the dendritic tree topology.
- Modular networks with separate sub-networks for soma, apical trunk, and \
  tuft compartments.
- Temporal scale separation with fast (Na/K) and slow (Ca/HCN) sub-networks.
- Oscillatory components (neural ODE with limit-cycle attractors) to capture \
  intrinsic resonance frequencies.

For each proposal, provide:
- architecture_type: a short identifier string
- description: one sentence explaining the core idea
- loss_recipe: dict with loss component names as keys and float weights as values
- hidden_dim: integer
- modifications: list of keyword strings (e.g. 'bottleneck', 'slow', 'continuous')
- rationale: why this might outperform existing approaches

Respond as a JSON array of objects. No commentary outside the JSON.
"""

SYSTEM_SURROGATE_GAP = """\
You are analysing a surrogate model search campaign for a Layer 5 Pyramidal \
Cell simulation and identifying coverage gaps in the search space.

Coverage summary so far:
{coverage_summary}

Best score achieved: {best_score}

Architecture families tested: {families}
Loss functions used: {losses}
Regularisation methods applied: {regularizers}
Hidden dimensions explored: {hidden_dims}

Identified gaps in coverage:
{gap_list}

Design 3-5 surrogate genome configurations that specifically fill these gaps. \
Each configuration must honour at least two of these five design principles:

1. Continuous-time dynamics -- use ODE-based or SDE-based layers to respect \
   the continuous nature of membrane voltage.
2. Embedded physics -- incorporate known biophysical constraints such as \
   reversal potentials, time-constant bounds, or conductance non-negativity.
3. Information bottleneck -- use a compressed latent space that forces the \
   surrogate to learn a low-dimensional dynamical manifold.
4. Biophysical supervision -- include auxiliary loss terms that supervise on \
   calcium concentration, dendritic voltage, or gate variables, not just \
   somatic voltage.
5. Oscillatory bias -- initialise or constrain components to produce \
   oscillatory solutions matching known intrinsic frequencies (theta, gamma).

For each configuration, provide:
- architecture_type: short identifier
- loss_recipe: dict of loss component name -> weight
- hidden_dim: integer
- modifications: list of keyword strings
- principles_used: list of which principles (by number) are honoured

Respond as a JSON array of objects. No commentary outside the JSON.
"""

SYSTEM_PROBE_BALLOON = """\
You are a neural data analyst designing novel probing strategies to decode \
biophysical variables from surrogate neural network hidden states.

Summary of probe results so far:
{probe_results_summary}

Probing methods already tried:
{methods_tried}

Variables that remain poorly decoded (zombie variables):
{zombie_variables}

Properties of these variables:
{variable_properties}

Your task: propose 3-5 genuinely novel probing strategies for extracting \
these zombie variables. Consider:

- Transformed coordinates: the information may be present in a non-linear \
  transformation of hidden states (e.g. log-conductance, gating-variable \
  products, energy-like quantities).
- Distributed across time: the variable may not be decodable from a single \
  snapshot but requires a temporal window or trajectory-level analysis.
- Dynamics not state: the information may be encoded in the rate of change \
  of hidden states rather than their instantaneous values.
- Superposition: the variable may be encoded as a linear combination of \
  many neurons, requiring sparse coding or dictionary learning.
- Regime-specific: encoding may differ across sub-threshold, spiking, and \
  bursting regimes, requiring conditional probes.
- Gate encoding: in LSTM-like architectures, information may live in \
  forget/input/output gate activations rather than the cell state.

For each strategy, provide:
- probe_type: short identifier
- description: one sentence explaining the approach
- transform: what transformation to apply to hidden states before probing
- temporal_mode: 'snapshot', 'window', or 'trajectory'
- decoder_type: what type of decoder to use
- target_variables: which zombie variables this strategy targets
- rationale: why this might succeed where previous probes failed

Respond as a JSON array of objects. No commentary outside the JSON.
"""

SYSTEM_PROBE_GAP = """\
You are analysing probe coverage across a surrogate model's hidden state \
space and identifying systematic gaps in the probing campaign.

Probing methods used: {methods}
Transforms applied: {transforms}
Temporal modes tested: {temporal_modes}
Target variables probed: {targets}
Conditions tested: {conditions}

Coverage matrix (method x transform x target):
{coverage_matrix}

Analyse this coverage matrix and identify:
1. Which (method, transform, target) combinations have NOT been tested.
2. Which combinations are most likely to reveal hidden information based on \
   the patterns in existing results.
3. Priority ordering: which gaps to fill first for maximum information gain.

For each recommended probe, provide:
- probe_type: short identifier
- transform: transformation to apply
- temporal_mode: temporal analysis mode
- decoder_type: decoder to use
- target_names: list of target variable names
- condition: optional condition string or null
- priority: integer 1-5 (1 = highest)
- rationale: why this gap matters

Respond as a JSON array of objects. No commentary outside the JSON.
"""


# ---------------------------------------------------------------------------
# LLMBalloonExpander
# ---------------------------------------------------------------------------

class LLMBalloonExpander:
    """Uses an LLM to propose novel surrogate architectures and probe strategies.

    Falls back gracefully when the anthropic SDK is not installed or no API
    key is provided -- all public methods return empty lists in that case.
    """

    def __init__(self, api_key: Optional[str] = None,
                 model: str = 'claude-sonnet-4-20250514'):
        self.api_key = api_key
        self.model = model
        self._client = None

    # -- public interface ---------------------------------------------------

    def propose_novel_surrogates(
        self,
        factory_state: dict,
        circuit_context: str,
    ) -> List[SurrogateGenome_v3]:
        """Ask the LLM to propose novel surrogate architectures.

        Parameters
        ----------
        factory_state : dict
            Dictionary with keys such as ``architectures_tested``,
            ``best_bio_score``, ``best_output_cc``, ``n_genomes``,
            ``verdict_distribution``.
        circuit_context : str
            Free-text description of the biophysical circuit being modelled.

        Returns
        -------
        list[SurrogateGenome_v3]
            Genome objects ready for evaluation.  Empty list on failure.
        """
        prompt = SYSTEM_SURROGATE_BALLOON.format(
            circuit_context=circuit_context,
            architectures_tested=factory_state.get('architectures_tested', '[]'),
            best_bio_score=factory_state.get('best_bio_score', 'N/A'),
            best_output_cc=factory_state.get('best_output_cc', 'N/A'),
            n_genomes=factory_state.get('n_genomes', 0),
            verdict_distribution=factory_state.get('verdict_distribution', '{}'),
        )
        raw = self._call_llm(prompt)
        return self._parse_proposals(raw)

    def analyze_gaps(
        self,
        factory_state: dict,
    ) -> List[SurrogateGenome_v3]:
        """Ask the LLM to fill coverage gaps in the search space.

        Parameters
        ----------
        factory_state : dict
            Dictionary with keys such as ``coverage_summary``,
            ``best_score``, ``families``, ``losses``, ``regularizers``,
            ``hidden_dims``, ``gap_list``.

        Returns
        -------
        list[SurrogateGenome_v3]
            Genome objects targeting identified gaps.  Empty list on failure.
        """
        prompt = SYSTEM_SURROGATE_GAP.format(
            coverage_summary=factory_state.get('coverage_summary', ''),
            best_score=factory_state.get('best_score', 'N/A'),
            families=factory_state.get('families', '[]'),
            losses=factory_state.get('losses', '[]'),
            regularizers=factory_state.get('regularizers', '[]'),
            hidden_dims=factory_state.get('hidden_dims', '[]'),
            gap_list=factory_state.get('gap_list', '[]'),
        )
        raw = self._call_llm(prompt)
        return self._parse_proposals(raw)

    # -- internal helpers ---------------------------------------------------

    def _parse_proposals(self, raw: str) -> List[SurrogateGenome_v3]:
        """Parse LLM JSON response into a list of genomes."""
        try:
            proposals = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse LLM response as JSON.")
            return []

        if not isinstance(proposals, list):
            logger.warning("LLM response is not a JSON array.")
            return []

        genomes: List[SurrogateGenome_v3] = []
        for proposal in proposals:
            try:
                genome = self._proposal_to_genome(proposal)
                genomes.append(genome)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not convert proposal to genome: %s", exc)
        return genomes

    def _proposal_to_genome(self, proposal: dict) -> SurrogateGenome_v3:
        """Convert a single LLM proposal dict into a SurrogateGenome_v3.

        Mapping rules
        -------------
        * ``architecture_type`` -> ``genome.architecture_type``
        * ``loss_recipe`` -> ``genome.loss_recipe``
        * ``hidden_dim`` -> ``genome.hidden_dim``
        * Modification keywords in ``modifications`` list:
          - ``'bottleneck'`` -> ``genome.use_information_bottleneck = True``
          - ``'slow'``       -> ``genome.use_slow_features = True``
          - ``'continuous'`` -> ``genome.continuous_time = True``
        """
        composer = SurrogateGenomeComposer()
        genome = composer.create_default()

        if 'architecture_type' in proposal:
            genome.architecture_type = str(proposal['architecture_type'])

        if 'loss_recipe' in proposal and isinstance(proposal['loss_recipe'], dict):
            genome.loss_recipe = {
                str(k): float(v) for k, v in proposal['loss_recipe'].items()
            }

        if 'hidden_dim' in proposal:
            genome.hidden_dim = int(proposal['hidden_dim'])

        modifications = proposal.get('modifications', [])
        if not isinstance(modifications, list):
            modifications = []

        for mod in modifications:
            mod_lower = str(mod).lower()
            if 'bottleneck' in mod_lower:
                genome.use_information_bottleneck = True
            if 'slow' in mod_lower:
                genome.use_slow_features = True
            if 'continuous' in mod_lower:
                genome.continuous_time = True

        return genome

    def _call_llm(self, prompt: str) -> str:
        """Call the Anthropic API.  Returns '[]' on any failure.

        Gracefully handles:
        * Missing ``anthropic`` package
        * Missing or invalid API key
        * Network / API errors
        """
        try:
            import anthropic  # noqa: F811
        except ImportError:
            logger.warning(
                "anthropic package is not installed. "
                "Install it with: pip install anthropic"
            )
            return '[]'

        if self.api_key is None:
            logger.warning("No API key provided; skipping LLM call.")
            return '[]'

        try:
            if self._client is None:
                self._client = anthropic.Anthropic(api_key=self.api_key)

            message = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            # Extract text from the response
            text = message.content[0].text if message.content else '[]'
            return text
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM call failed: %s", exc)
            return '[]'
