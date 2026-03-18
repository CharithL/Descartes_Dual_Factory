"""
DESCARTES Dual Factory v3.0 -- C2 Outer Loop: Surrogate Factory

The SurrogateFactory is the outer evolutionary loop.  It evolves
surrogate model genomes through four phases:

  Phase 1 (0-25%):   Random template exploration
  Phase 2 (25-50%):  Mutation + crossover of best performers
  Phase 3 (50-75%):  DreamCoder library-guided composition
  Phase 4 (75-100%): LLM balloon for creative architecture proposals

Each round: select genome -> train + validate -> C1 inner loop probing ->
fitness scoring -> Thompson sampling update -> DreamCoder wake every 25 rounds.
"""

import logging
import time
from collections import defaultdict

import numpy as np

from .surrogate_genome import SurrogateGenome_v3, SurrogateGenomeComposer
from .config import (
    THOMPSON_ALPHA_PRIOR,
    THOMPSON_BETA_PRIOR,
    DREAMCODER_WAKE_INTERVAL,
    PHASE_1_END,
    PHASE_2_END,
    PHASE_3_END,
    ARCHITECTURES,
)

logger = logging.getLogger(__name__)


class SurrogateFactory:
    """C2 outer loop: evolutionary campaign over surrogate architectures.

    Parameters
    ----------
    circuit_data : dict
        Training/validation/test data for the circuit.
    circuit_context : dict
        Metadata about the circuit (neuron type, protocols, etc.).
    probe_factory : ProbingFactoryEvaluator
        The C1 inner loop evaluator instance.
    fitness_fn : callable or None
        Custom fitness function.  If None, uses default weighted fitness.
    """

    def __init__(self, circuit_data, circuit_context, probe_factory,
                 fitness_fn=None):
        self.circuit_data = circuit_data
        self.circuit_context = circuit_context
        self.probe_factory = probe_factory
        self.fitness_fn = fitness_fn

        # Composition tools
        self.composer = SurrogateGenomeComposer()
        self.trainer = None      # SurrogateTrainer — set externally or lazily
        self.dreamcoder = None   # DreamCoder — set externally or lazily
        self.llm_balloon = None  # LLMBalloon — set externally or lazily

        # Thompson sampling state: per-architecture Beta posteriors
        self.arch_successes = {
            arch: THOMPSON_ALPHA_PRIOR for arch in ARCHITECTURES}
        self.arch_failures = {
            arch: THOMPSON_BETA_PRIOR for arch in ARCHITECTURES}

        # Campaign history
        self.results = []
        self.best_fitness = -np.inf
        self.best_genome = None
        self.best_probing = None
        self.stale_rounds = 0

        # Random state
        self.rng = np.random.default_rng(42)

    # ── Main campaign loop ─────────────────────────────────────────────

    def run_campaign(self, n_rounds=200, device='cpu'):
        """Run the full evolutionary campaign.

        Parameters
        ----------
        n_rounds : int
            Total number of evolution rounds.
        device : str
            Torch device string.

        Returns
        -------
        dict
            Final campaign report from _final_report().
        """
        logger.info("Starting surrogate factory campaign: %d rounds", n_rounds)
        start_time = time.time()

        for round_idx in range(n_rounds):
            logger.info("=== Round %d / %d ===", round_idx + 1, n_rounds)

            # Phase 1: Select genome
            genome = self._select_genome(round_idx, n_rounds)
            genome.genome_id = genome.fingerprint()
            logger.info("Selected genome: arch=%s, hidden=%d, id=%s",
                        genome.architecture, genome.hidden_dim,
                        genome.genome_id[:12])

            # Phase 2: Train and validate
            if self.trainer is not None:
                train_result = self.trainer.train(
                    genome, self.circuit_data, device=device)
                model = train_result.get('model', None)
                val_loss = train_result.get('val_loss', np.inf)
            else:
                logger.warning("No trainer configured; skipping training")
                train_result = {'val_loss': np.inf}
                model = None
                val_loss = np.inf

            # Phase 3: C1 inner loop probing (if model trained)
            probing_result = None
            if model is not None:
                probing_result = self.probe_factory.evaluate_surrogate(
                    model, self.circuit_data, device=device)
                logger.info("Probing verdict: %s (tier %d, %d mandatory)",
                            probing_result['verdict'],
                            probing_result['tier_reached'],
                            probing_result['n_mandatory_variables'])

            # Phase 4: Compute fitness
            if self.fitness_fn is not None:
                fitness = self.fitness_fn(
                    genome, train_result, probing_result)
            else:
                fitness = self._default_fitness(
                    train_result, probing_result)

            logger.info("Fitness: %.4f (best so far: %.4f)",
                        fitness, self.best_fitness)

            # Update best
            is_improvement = fitness > self.best_fitness
            if is_improvement:
                self.best_fitness = fitness
                self.best_genome = genome
                self.best_probing = probing_result
                self.stale_rounds = 0
            else:
                self.stale_rounds += 1

            # Thompson sampling update
            is_success = fitness > 0.5  # Threshold for "success"
            self._update_thompson(genome.architecture, is_success)

            # Record result
            self._record_result(genome, train_result, fitness, round_idx)

            # DreamCoder wake phase every DREAMCODER_WAKE_INTERVAL rounds
            if (self.dreamcoder is not None and
                    (round_idx + 1) % DREAMCODER_WAKE_INTERVAL == 0):
                logger.info("DreamCoder wake phase at round %d", round_idx + 1)
                self.dreamcoder.wake(self.results)

            # Progress report
            self._report_progress(round_idx)

        elapsed = time.time() - start_time
        logger.info("Campaign complete in %.1f seconds", elapsed)

        return self._final_report()

    # ── Genome selection ───────────────────────────────────────────────

    def _select_genome(self, round_idx, n_rounds):
        """Select a genome based on the current campaign phase.

        Phase 1 (0 - PHASE_1_END):   Random template exploration
        Phase 2 (PHASE_1_END - PHASE_2_END):  Mutation/crossover of best
        Phase 3 (PHASE_2_END - PHASE_3_END):  DreamCoder library
        Phase 4 (PHASE_3_END - 1.0):          LLM balloon

        Parameters
        ----------
        round_idx : int
            Current round index.
        n_rounds : int
            Total number of rounds.

        Returns
        -------
        SurrogateGenome_v3
            The selected genome for this round.
        """
        progress = round_idx / max(n_rounds, 1)

        # Phase 1: Random exploration
        if progress < PHASE_1_END:
            # Thompson-sampled architecture + random hyperparameters
            arch = self._thompson_sample_architecture()
            genome = self.composer.compose_random(self.rng)
            genome.architecture = arch
            return genome

        # Phase 2: Mutation + crossover of best performers
        if progress < PHASE_2_END:
            if len(self.results) < 2 or self.best_genome is None:
                return self.composer.compose_random(self.rng)

            # Pick two parents from top quartile
            sorted_results = sorted(
                self.results, key=lambda r: r['fitness'], reverse=True)
            top_k = max(2, len(sorted_results) // 4)
            top_genomes = [r['genome'] for r in sorted_results[:top_k]]

            if self.rng.random() < 0.5:
                # Mutation
                parent = top_genomes[int(self.rng.integers(0, len(top_genomes)))]
                return self.composer.mutate(parent, rng=self.rng)
            else:
                # Crossover
                idx = self.rng.choice(len(top_genomes), size=2, replace=False)
                return self.composer.crossover(
                    top_genomes[idx[0]], top_genomes[idx[1]], rng=self.rng)

        # Phase 3: DreamCoder library
        if progress < PHASE_3_END:
            if self.dreamcoder is not None and self.dreamcoder.has_library():
                genome = self.dreamcoder.sample_from_library(self.rng)
                if genome is not None:
                    return genome
            # Fallback: mutate best
            if self.best_genome is not None:
                return self.composer.mutate(self.best_genome, rng=self.rng)
            return self.composer.compose_random(self.rng)

        # Phase 4: LLM balloon
        if self.llm_balloon is not None and self.stale_rounds > 0:
            state = self._get_factory_state()
            genome = self.llm_balloon.propose_genome(state)
            if genome is not None:
                return genome
        # Fallback: mutate best
        if self.best_genome is not None:
            return self.composer.mutate(self.best_genome, rng=self.rng)
        return self.composer.compose_random(self.rng)

    # ── Thompson sampling ──────────────────────────────────────────────

    def _thompson_sample_architecture(self):
        """Sample an architecture using Thompson sampling (Beta posteriors).

        Returns
        -------
        str
            Architecture name.
        """
        samples = {}
        for arch in ARCHITECTURES:
            samples[arch] = self.rng.beta(
                self.arch_successes[arch],
                self.arch_failures[arch])
        return max(samples, key=samples.get)

    def _update_thompson(self, architecture, is_success):
        """Update Thompson sampling posteriors for an architecture.

        Parameters
        ----------
        architecture : str
            The architecture that was evaluated.
        is_success : bool
            Whether this round was a success.
        """
        if architecture not in self.arch_successes:
            self.arch_successes[architecture] = THOMPSON_ALPHA_PRIOR
            self.arch_failures[architecture] = THOMPSON_BETA_PRIOR

        if is_success:
            self.arch_successes[architecture] += 1
        else:
            self.arch_failures[architecture] += 1

    # ── Recording and reporting ────────────────────────────────────────

    def _record_result(self, genome, train_result, fitness, round_idx):
        """Record a single round result.

        Parameters
        ----------
        genome : SurrogateGenome_v3
            The genome that was evaluated.
        train_result : dict
            Training results.
        fitness : float
            Computed fitness score.
        round_idx : int
            Current round index.
        """
        self.results.append({
            'round': round_idx,
            'genome': genome,
            'genome_id': genome.genome_id,
            'architecture': genome.architecture,
            'hidden_dim': genome.hidden_dim,
            'train_result': train_result,
            'fitness': fitness,
            'timestamp': time.time(),
        })

    def _get_factory_state(self):
        """Get current factory state for LLM balloon context.

        Returns
        -------
        dict
            Summary of campaign state for the LLM to reason about.
        """
        return {
            'n_rounds_completed': len(self.results),
            'best_fitness': self.best_fitness,
            'best_genome': self.best_genome.to_dict() if self.best_genome else None,
            'stale_rounds': self.stale_rounds,
            'thompson_posteriors': {
                arch: {
                    'successes': self.arch_successes[arch],
                    'failures': self.arch_failures[arch],
                }
                for arch in ARCHITECTURES
            },
            'coverage': self._coverage_summary(),
            'gaps': self._detect_gaps(),
            'regularizers': self._regularizers_summary(),
        }

    def _coverage_summary(self):
        """Summarise which architectures have been explored.

        Returns
        -------
        dict
            Mapping from architecture to count of evaluations.
        """
        counts = defaultdict(int)
        for r in self.results:
            counts[r['architecture']] += 1
        return dict(counts)

    def _detect_gaps(self):
        """Detect under-explored regions of the search space.

        Returns
        -------
        list[str]
            List of architecture names with zero or very few evaluations.
        """
        coverage = self._coverage_summary()
        threshold = max(1, len(self.results) // (2 * len(ARCHITECTURES)))
        gaps = [
            arch for arch in ARCHITECTURES
            if coverage.get(arch, 0) < threshold
        ]
        return gaps

    def _regularizers_summary(self):
        """Summarise regularizer usage across the campaign.

        Returns
        -------
        dict
            Statistics on regularizer choices.
        """
        reg_stats = defaultdict(list)
        for r in self.results:
            g = r['genome']
            reg_stats['weight_decay'].append(g.weight_decay)
            reg_stats['hidden_l1'].append(g.hidden_l1)
            reg_stats['ib_used'].append(g.information_bottleneck)
            reg_stats['slow_feature'].append(g.slow_feature_penalty)
            reg_stats['disentanglement'].append(g.disentanglement_penalty)

        summary = {}
        for key, values in reg_stats.items():
            if isinstance(values[0], bool):
                summary[key] = {'fraction_true': np.mean(values)}
            else:
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
        return summary

    def _report_progress(self, round_idx):
        """Log a progress report for the current round.

        Parameters
        ----------
        round_idx : int
            Current round index.
        """
        n_done = round_idx + 1
        n_total = len(self.results)
        logger.info(
            "Progress: round %d | total evaluated: %d | "
            "best fitness: %.4f | stale: %d | "
            "best arch: %s",
            n_done, n_total, self.best_fitness, self.stale_rounds,
            self.best_genome.architecture if self.best_genome else 'none')

    def _final_report(self):
        """Generate the final campaign report.

        Returns
        -------
        dict
            Comprehensive summary of the campaign.
        """
        # Sort results by fitness
        sorted_results = sorted(
            self.results, key=lambda r: r['fitness'], reverse=True)

        # Top 10 genomes
        top_10 = []
        for r in sorted_results[:10]:
            top_10.append({
                'genome_id': r['genome_id'],
                'architecture': r['architecture'],
                'hidden_dim': r['hidden_dim'],
                'fitness': r['fitness'],
                'round': r['round'],
            })

        # Architecture performance summary
        arch_fitness = defaultdict(list)
        for r in self.results:
            arch_fitness[r['architecture']].append(r['fitness'])

        arch_summary = {}
        for arch, fitnesses in arch_fitness.items():
            arch_summary[arch] = {
                'n_evaluated': len(fitnesses),
                'mean_fitness': float(np.mean(fitnesses)),
                'max_fitness': float(np.max(fitnesses)),
                'std_fitness': float(np.std(fitnesses)),
            }

        return {
            'n_rounds': len(self.results),
            'best_fitness': self.best_fitness,
            'best_genome': self.best_genome.to_dict() if self.best_genome else None,
            'best_probing': self.best_probing,
            'top_10': top_10,
            'architecture_summary': arch_summary,
            'thompson_posteriors': {
                arch: {
                    'successes': self.arch_successes[arch],
                    'failures': self.arch_failures[arch],
                }
                for arch in ARCHITECTURES
            },
            'coverage': self._coverage_summary(),
            'gaps': self._detect_gaps(),
            'regularizers': self._regularizers_summary(),
        }

    # ── Default fitness function ───────────────────────────────────────

    @staticmethod
    def _default_fitness(train_result, probing_result):
        """Default weighted fitness: output accuracy + bio correspondence + causal.

        Parameters
        ----------
        train_result : dict
            Must contain 'val_loss'.
        probing_result : dict or None
            Results from C1 inner loop.

        Returns
        -------
        float
            Fitness score in [0, 1].
        """
        from .config import FITNESS_ALPHA, FITNESS_BETA, FITNESS_GAMMA

        # Output accuracy component (convert loss to score)
        val_loss = train_result.get('val_loss', np.inf)
        if np.isinf(val_loss) or np.isnan(val_loss):
            accuracy_score = 0.0
        else:
            accuracy_score = 1.0 / (1.0 + val_loss)

        # Biological correspondence component
        bio_score = 0.0
        causal_score = 0.0

        if probing_result is not None:
            # Count encoded variables
            per_var = probing_result.get('per_variable', {})
            n_vars = max(len(per_var), 1)
            n_encoded = sum(
                1 for v in per_var.values()
                if v.get('hardened_verdict', '') in
                {'CONFIRMED_ENCODED', 'CANDIDATE_ENCODED', 'NONLINEAR_ENCODED'}
            )
            bio_score = n_encoded / n_vars

            # Causal component
            n_mandatory = probing_result.get('n_mandatory_variables', 0)
            causal_score = n_mandatory / n_vars

        fitness = (
            FITNESS_ALPHA * accuracy_score
            + FITNESS_BETA * bio_score
            + FITNESS_GAMMA * causal_score
        )
        return float(np.clip(fitness, 0.0, 1.0))
