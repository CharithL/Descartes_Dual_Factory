"""Surrogate fitness evaluation for the DESCARTES Dual Factory v3.0.

Computes a composite fitness score from three weighted components:
  * **Output accuracy** (alpha)  -- cross-condition correlation.
  * **Biological correspondence** (beta)  -- zombie-probing verdict.
  * **Causal necessity** (gamma)  -- mandatory feature count.
"""


class SurrogateFitness:
    """Weighted fitness computation for surrogate genomes."""

    # Mapping from probing verdict string to biological plausibility score.
    BIO_SCORE_MAP: dict = {
        "MANDATORY": 1.0,
        "CONFIRMED_NON_ZOMBIE": 0.85,
        "SUPERPOSED_NON_ZOMBIE": 0.75,
        "NONLINEAR_ENCODED": 0.65,
        "CANDIDATE_ENCODED": 0.5,
        "AMBIGUOUS": 0.3,
        "LIKELY_ZOMBIE": 0.1,
        "SPURIOUS_DRIFT": 0.05,
        "CONFIRMED_ZOMBIE": 0.0,
    }

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.5,
        gamma: float = 0.2,
    ) -> None:
        """
        Parameters
        ----------
        alpha : float
            Weight for output accuracy score.
        beta : float
            Weight for biological correspondence score.
        gamma : float
            Weight for causal necessity score.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute(
        self,
        output_result: dict,
        probing_verdict: dict,
    ) -> dict:
        """Compute the composite fitness for one surrogate genome.

        Parameters
        ----------
        output_result : dict
            Must contain ``'output_cc'`` (float) -- the cross-condition
            correlation from surrogate training.
        probing_verdict : dict
            Must contain ``'verdict'`` (str) and ``'n_mandatory'`` (int).

        Returns
        -------
        dict
            Keys: fitness, output_score, bio_score, causal_score, verdict.
        """
        # -- Output accuracy score ----------------------------------------
        cc = output_result.get("output_cc", 0.0)
        output_score = min(1.0, cc / 0.95)

        # -- Biological correspondence score ------------------------------
        verdict_str = probing_verdict.get("verdict", "AMBIGUOUS")
        bio_score = self.BIO_SCORE_MAP.get(verdict_str, 0.0)

        # -- Causal necessity score ---------------------------------------
        n_mandatory = probing_verdict.get("n_mandatory", 0)
        causal_score = min(1.0, n_mandatory / 5.0)

        # -- Composite fitness --------------------------------------------
        fitness = (
            self.alpha * output_score
            + self.beta * bio_score
            + self.gamma * causal_score
        )

        return {
            "fitness": fitness,
            "output_score": output_score,
            "bio_score": bio_score,
            "causal_score": causal_score,
            "verdict": verdict_str,
        }
