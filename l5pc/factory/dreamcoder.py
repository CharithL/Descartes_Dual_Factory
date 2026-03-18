"""SurrogateDreamCoder -- wake/sleep library learning for surrogate genomes.

Inspired by the DreamCoder program synthesis system, this module maintains a
library of reusable architectural patterns discovered from successful surrogate
genomes.  The wake phase extracts patterns from top performers; the sleep phase
prunes weak patterns and (in future) discovers novel combinations.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from .surrogate_genome import SurrogateGenome_v3, SurrogateGenomeComposer

logger = logging.getLogger(__name__)


class SurrogateDreamCoder:
    """Wake/sleep library learning for surrogate genome patterns.

    Maintains a library of reusable patterns extracted from high-fitness
    genomes and uses them to compose new candidates via weighted sampling.
    """

    def __init__(self):
        self.library: Dict[str, dict] = {}
        self.pattern_scores: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Wake phase
    # ------------------------------------------------------------------

    def wake_phase(self, campaign_results: List[dict]) -> None:
        """Extract patterns from top-performing genomes.

        Parameters
        ----------
        campaign_results : list[dict]
            Each entry must contain ``'genome'`` (a SurrogateGenome_v3) and
            ``'fitness'`` (a float score).
        """
        if not campaign_results:
            return

        # Sort descending by fitness and keep top 20 %
        sorted_results = sorted(
            campaign_results, key=lambda r: r.get('fitness', 0.0), reverse=True
        )
        n_top = max(1, len(sorted_results) // 5)
        top_results = sorted_results[:n_top]

        top_genomes = [r['genome'] for r in top_results if 'genome' in r]
        top_fitnesses = [r.get('fitness', 0.0) for r in top_results if 'genome' in r]

        if not top_genomes:
            return

        patterns = self._extract_patterns(top_genomes)

        for pat_name, pat_value in patterns.items():
            self.library[pat_name] = pat_value
            # Record the mean fitness of the cohort that produced this pattern
            mean_fit = float(np.mean(top_fitnesses)) if top_fitnesses else 0.0
            self.pattern_scores[pat_name].append(mean_fit)

        logger.info(
            "Wake phase: extracted %d patterns from %d top genomes.",
            len(patterns), len(top_genomes),
        )

    # ------------------------------------------------------------------
    # Sleep phase
    # ------------------------------------------------------------------

    def sleep_phase(self) -> None:
        """Prune weak patterns and discover new combinations.

        A pattern is pruned if it has at least 5 fitness samples and its
        mean fitness is below 0.2.
        """
        to_remove = []
        for pat_name, scores in self.pattern_scores.items():
            if len(scores) >= 5 and float(np.mean(scores)) < 0.2:
                to_remove.append(pat_name)

        for pat_name in to_remove:
            self.library.pop(pat_name, None)
            self.pattern_scores.pop(pat_name, None)
            logger.info("Sleep phase: pruned weak pattern '%s'.", pat_name)

        self._discover_combination_patterns()

    # ------------------------------------------------------------------
    # Compose from library
    # ------------------------------------------------------------------

    def compose_from_library(
        self, rng: np.random.Generator
    ) -> SurrogateGenome_v3:
        """Create a new genome by sampling and applying library patterns.

        Patterns are selected with probability proportional to their average
        fitness score.  Between 1 and 3 patterns are applied to a fresh
        default genome.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator for reproducibility.

        Returns
        -------
        SurrogateGenome_v3
            A newly composed genome.
        """
        composer = SurrogateGenomeComposer()
        genome = composer.create_default()

        if not self.library:
            return genome

        # Build weight vector from average fitness
        pat_names = list(self.library.keys())
        weights = np.array([
            float(np.mean(self.pattern_scores[name]))
            if self.pattern_scores[name] else 0.0
            for name in pat_names
        ])

        # Ensure non-negative weights; add small epsilon to avoid all-zero
        weights = np.maximum(weights, 0.0) + 1e-8
        weights /= weights.sum()

        n_patterns = int(rng.integers(1, min(4, len(pat_names) + 1)))
        chosen_indices = rng.choice(
            len(pat_names), size=n_patterns, replace=False, p=weights
        )

        for idx in chosen_indices:
            pat_name = pat_names[idx]
            pattern = self.library[pat_name]
            genome = self._apply_pattern(genome, pattern)

        return genome

    # ------------------------------------------------------------------
    # Pattern extraction
    # ------------------------------------------------------------------

    def _extract_patterns(self, genomes: List[SurrogateGenome_v3]) -> dict:
        """Extract recurring patterns from a list of high-fitness genomes.

        Heuristic rules
        ---------------
        * Architecture clustering >= 40 % -> architecture pattern
        * ``use_information_bottleneck`` True in >= 50 % -> IB pattern
        * Bio-loss present in >= 50 % of loss recipes -> bio_loss pattern
        * ``use_slow_features`` True in >= 40 % -> slow_features pattern
        * Hidden-dim clustering (most common value appears >= 40 %) -> dim pattern
        """
        if not genomes:
            return {}

        n = len(genomes)
        patterns: dict = {}

        # --- Architecture clustering ---
        arch_counts: Dict[str, int] = defaultdict(int)
        for g in genomes:
            arch_counts[getattr(g, 'architecture_type', 'unknown')] += 1
        for arch, count in arch_counts.items():
            if count / n >= 0.4:
                pat_key = f"arch_{arch}"
                patterns[pat_key] = {'architecture_type': arch}

        # --- Information bottleneck ---
        ib_count = sum(
            1 for g in genomes if getattr(g, 'use_information_bottleneck', False)
        )
        if ib_count / n >= 0.5:
            patterns['use_information_bottleneck'] = {
                'use_information_bottleneck': True
            }

        # --- Bio-loss ---
        bio_loss_count = sum(
            1 for g in genomes
            if any(
                'bio' in str(k).lower()
                for k in getattr(g, 'loss_recipe', {}).keys()
            )
        )
        if bio_loss_count / n >= 0.5:
            # Collect the most common bio-loss configuration
            bio_recipes: Dict[str, List[float]] = defaultdict(list)
            for g in genomes:
                for k, v in getattr(g, 'loss_recipe', {}).items():
                    if 'bio' in str(k).lower():
                        bio_recipes[k].append(float(v))
            representative = {
                k: float(np.mean(vs)) for k, vs in bio_recipes.items()
            }
            patterns['bio_loss'] = {'loss_recipe_update': representative}

        # --- Slow features ---
        slow_count = sum(
            1 for g in genomes if getattr(g, 'use_slow_features', False)
        )
        if slow_count / n >= 0.4:
            patterns['use_slow_features'] = {'use_slow_features': True}

        # --- Hidden-dim clustering ---
        dim_counts: Dict[int, int] = defaultdict(int)
        for g in genomes:
            dim_counts[getattr(g, 'hidden_dim', 64)] += 1
        for dim, count in dim_counts.items():
            if count / n >= 0.4:
                patterns[f"hidden_dim_{dim}"] = {'hidden_dim': dim}

        return patterns

    # ------------------------------------------------------------------
    # Pattern application / matching
    # ------------------------------------------------------------------

    def _apply_pattern(
        self, genome: SurrogateGenome_v3, pattern: dict
    ) -> SurrogateGenome_v3:
        """Apply a pattern dictionary to a genome via setattr.

        Special key ``loss_recipe_update`` merges into the existing loss
        recipe rather than replacing it outright.
        """
        for key, value in pattern.items():
            if key == 'loss_recipe_update' and isinstance(value, dict):
                existing = getattr(genome, 'loss_recipe', {}) or {}
                existing.update(value)
                genome.loss_recipe = existing
            elif hasattr(genome, key):
                setattr(genome, key, value)
        return genome

    def _genome_matches_pattern(
        self, genome: SurrogateGenome_v3, pattern: dict
    ) -> bool:
        """Return True if *genome* already exhibits *pattern*."""
        for key, value in pattern.items():
            if key == 'loss_recipe_update':
                recipe = getattr(genome, 'loss_recipe', {}) or {}
                if not all(k in recipe for k in value):
                    return False
            else:
                if getattr(genome, key, None) != value:
                    return False
        return True

    # ------------------------------------------------------------------
    # Placeholder for future combination discovery
    # ------------------------------------------------------------------

    def _discover_combination_patterns(self) -> None:
        """Discover novel pattern combinations during the sleep phase.

        Placeholder for future implementation -- will analyse pairwise
        co-occurrence of patterns in high-fitness genomes and propose
        merged super-patterns.
        """
        pass
