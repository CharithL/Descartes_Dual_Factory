"""
DESCARTES Dual Factory v3.0 -- Dual Factory Orchestrator

The DualFactoryOrchestrator ties together the C1 inner loop
(ProbingFactoryEvaluator) and the C2 outer loop (SurrogateFactory)
into a single end-to-end campaign.

After the campaign completes, it also provides post-hoc analysis:
  - Architecture clustering via BayesianGaussianMixture (Dirichlet process)
  - Design pattern extraction from the DreamCoder library
"""

import logging
from collections import defaultdict

import numpy as np
from sklearn.mixture import BayesianGaussianMixture

from .probing_evaluator import ProbingFactoryEvaluator
from .surrogate_factory import SurrogateFactory

logger = logging.getLogger(__name__)


class DualFactoryOrchestrator:
    """End-to-end orchestrator for the DESCARTES dual factory.

    Creates both the C1 (probing evaluator) and C2 (surrogate factory)
    loops and runs the full evolutionary campaign.

    Parameters
    ----------
    circuit_data : dict
        Training/validation/test data for the circuit.
    circuit_context : dict
        Metadata about the circuit (neuron type, protocols, etc.).
    bio_targets : dict[str, np.ndarray]
        Mapping from variable name to (N_samples,) target array.
    target_names : list[str]
        Ordered list of biological variable names.
    """

    def __init__(self, circuit_data, circuit_context, bio_targets,
                 target_names):
        self.circuit_data = circuit_data
        self.circuit_context = circuit_context
        self.bio_targets = bio_targets
        self.target_names = list(target_names)

        # C1 inner loop
        self.probe_evaluator = ProbingFactoryEvaluator(
            bio_targets=bio_targets,
            target_names=target_names,
        )

        # C2 outer loop
        self.surrogate_factory = SurrogateFactory(
            circuit_data=circuit_data,
            circuit_context=circuit_context,
            probe_factory=self.probe_evaluator,
        )

    # ── Main entry point ───────────────────────────────────────────────

    def run_full_campaign(self, n_rounds=200, device='cpu'):
        """Run the full dual factory campaign.

        Parameters
        ----------
        n_rounds : int
            Number of evolutionary rounds.
        device : str
            Torch device string.

        Returns
        -------
        dict
            Campaign results including final report, architecture
            clusters, and design patterns.
        """
        logger.info("Starting DualFactoryOrchestrator campaign: "
                    "%d rounds, %d bio targets",
                    n_rounds, len(self.target_names))

        # Run the C2 outer loop (which internally calls C1)
        factory_report = self.surrogate_factory.run_campaign(
            n_rounds=n_rounds, device=device)

        # Post-hoc analysis
        clusters = self._cluster_architectures()
        self._generate_design_patterns()

        results = {
            'factory_report': factory_report,
            'architecture_clusters': clusters,
        }

        logger.info("DualFactoryOrchestrator campaign complete")
        return results

    # ── Post-hoc analysis ──────────────────────────────────────────────

    def _cluster_architectures(self):
        """Cluster evaluated architectures using BayesianGaussianMixture.

        Uses a Dirichlet process prior (weight_concentration_prior_type=
        'dirichlet_process') so the number of clusters is inferred
        automatically from the data.

        Returns
        -------
        dict
            cluster_labels : np.ndarray
                Cluster assignment for each evaluated genome.
            n_clusters : int
                Number of effective clusters found.
            cluster_summaries : list[dict]
                Per-cluster statistics (mean fitness, architectures, etc.).
        """
        results = self.surrogate_factory.results
        if len(results) < 3:
            logger.warning("Too few results (%d) for clustering", len(results))
            return {
                'cluster_labels': np.array([]),
                'n_clusters': 0,
                'cluster_summaries': [],
            }

        # Build feature vectors from genome parameters
        feature_vectors = []
        for r in results:
            g = r['genome']
            vec = [
                g.hidden_dim / 512.0,             # Normalised hidden dim
                g.n_layers / 4.0,                  # Normalised layers
                g.dropout,                         # Already 0-1
                np.log10(g.learning_rate + 1e-8),  # Log learning rate
                g.weight_decay * 1000,             # Scaled weight decay
                g.hidden_l1 * 1000,                # Scaled L1
                float(g.information_bottleneck),   # Binary
                g.slow_feature_penalty,
                g.disentanglement_penalty,
                float(g.continuous_time),
                float(g.compartmental_structure),
                r['fitness'],                      # Fitness as a feature
            ]
            feature_vectors.append(vec)

        X = np.array(feature_vectors)

        # Dirichlet process Gaussian mixture
        bgm = BayesianGaussianMixture(
            n_components=min(20, len(X)),
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.1,
            max_iter=500,
            random_state=42,
        )
        labels = bgm.fit_predict(X)

        # Count effective clusters (those with at least 1 assignment)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Per-cluster summaries
        cluster_summaries = []
        for cl in unique_labels:
            mask = labels == cl
            cl_results = [r for r, m in zip(results, mask) if m]
            cl_fitnesses = [r['fitness'] for r in cl_results]
            cl_archs = [r['architecture'] for r in cl_results]

            # Most common architecture in cluster
            arch_counts = defaultdict(int)
            for a in cl_archs:
                arch_counts[a] += 1
            dominant_arch = max(arch_counts, key=arch_counts.get)

            cluster_summaries.append({
                'cluster_id': int(cl),
                'n_members': int(mask.sum()),
                'mean_fitness': float(np.mean(cl_fitnesses)),
                'max_fitness': float(np.max(cl_fitnesses)),
                'dominant_architecture': dominant_arch,
                'architecture_distribution': dict(arch_counts),
            })

        logger.info("Architecture clustering: %d effective clusters from "
                    "%d evaluations", n_clusters, len(results))

        return {
            'cluster_labels': labels,
            'n_clusters': n_clusters,
            'cluster_summaries': cluster_summaries,
        }

    def _generate_design_patterns(self):
        """Print design patterns extracted from the DreamCoder library.

        If the DreamCoder module has been attached and has accumulated
        library entries, this prints them as reusable design patterns.
        """
        dc = self.surrogate_factory.dreamcoder
        if dc is None:
            logger.info("No DreamCoder module attached; "
                        "skipping design pattern generation")
            return

        if not hasattr(dc, 'library') or not dc.library:
            logger.info("DreamCoder library is empty; "
                        "no design patterns to report")
            return

        print("\n" + "=" * 70)
        print("DESIGN PATTERNS FROM DREAMCODER LIBRARY")
        print("=" * 70)

        for i, entry in enumerate(dc.library):
            pattern_name = entry.get('name', 'pattern_%d' % i)
            pattern_fitness = entry.get('mean_fitness', 0.0)
            pattern_count = entry.get('count', 0)
            pattern_genome = entry.get('genome_template', {})

            print("\nPattern %d: %s" % (i + 1, pattern_name))
            print("  Mean fitness: %.4f" % pattern_fitness)
            print("  Occurrences:  %d" % pattern_count)

            if isinstance(pattern_genome, dict):
                arch = pattern_genome.get('architecture', 'unknown')
                hdim = pattern_genome.get('hidden_dim', '?')
                print("  Architecture: %s (hidden=%s)" % (arch, hdim))

                # Print notable regularizer choices
                if pattern_genome.get('information_bottleneck'):
                    print("  Information bottleneck: ON")
                if pattern_genome.get('slow_feature_penalty', 0) > 0:
                    print("  Slow feature penalty: %.4f" %
                          pattern_genome['slow_feature_penalty'])
                if pattern_genome.get('disentanglement_penalty', 0) > 0:
                    print("  Disentanglement penalty: %.4f" %
                          pattern_genome['disentanglement_penalty'])

        print("\n" + "=" * 70)
