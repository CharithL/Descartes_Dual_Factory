"""Zombie verdict generator for the DESCARTES Dual Factory v3.0.

Classifies hidden-state features into one of nine verdict categories
using a priority-ordered evidence cascade:

    MANDATORY > SUPERPOSED_NON_ZOMBIE > NONLINEAR_ENCODED >
    SPURIOUS_DRIFT > CONFIRMED_NON_ZOMBIE > CANDIDATE_ENCODED >
    CONFIRMED_ZOMBIE > LIKELY_ZOMBIE > AMBIGUOUS
"""

from typing import Any, Dict


class ZombieVerdictGenerator_v3:
    """Generate zombie/non-zombie verdicts from an evidence bundle."""

    # Verdict types in strict priority order (highest first).
    VERDICT_PRIORITY = [
        "MANDATORY",
        "SUPERPOSED_NON_ZOMBIE",
        "NONLINEAR_ENCODED",
        "SPURIOUS_DRIFT",
        "CONFIRMED_NON_ZOMBIE",
        "CANDIDATE_ENCODED",
        "CONFIRMED_ZOMBIE",
        "LIKELY_ZOMBIE",
        "AMBIGUOUS",
    ]

    def generate_verdict(self, evidence_bundle: Dict[str, Any]) -> dict:
        """Classify a hidden feature based on multi-method evidence.

        Parameters
        ----------
        evidence_bundle : dict
            Expected keys (all optional -- missing evidence is treated as
            absent/failing):

            * ``resample_ablation`` : dict with ``'causal'`` (bool).
            * ``sae_superposition`` : dict with ``'detected'`` (bool).
            * ``ridge_r2``          : float -- ridge probe R^2.
            * ``mlp_r2``            : float -- MLP probe R^2.
            * ``frequency_drift``   : dict with ``'detected'`` (bool).
            * ``significance``      : dict with ``'p_value'`` (float).
            * ``positive_methods``  : list[str] -- names of methods that
              detected encoding.
            * ``tost``              : dict with ``'equivalent'`` (bool).
            * ``bf01``              : float -- Bayes factor (null / alt).

        Returns
        -------
        dict
            Keys: verdict (str), confidence (float), evidence (dict).
        """
        # -- Convenience extractors ---------------------------------------
        resample = evidence_bundle.get("resample_ablation", {})
        sae = evidence_bundle.get("sae_superposition", {})
        ridge_r2 = evidence_bundle.get("ridge_r2", 0.0)
        mlp_r2 = evidence_bundle.get("mlp_r2", 0.0)
        freq_drift = evidence_bundle.get("frequency_drift", {})
        significance = evidence_bundle.get("significance", {})
        positive_methods = evidence_bundle.get("positive_methods", [])
        tost = evidence_bundle.get("tost", {})
        bf01 = evidence_bundle.get("bf01", 1.0)

        causal = resample.get("causal", False)
        sae_detected = sae.get("detected", False)
        ridge_low = ridge_r2 < 0.05
        ridge_high = ridge_r2 >= 0.2
        mlp_high = mlp_r2 >= 0.2
        freq_detected = freq_drift.get("detected", False)
        p_value = significance.get("p_value", 1.0)
        sig = p_value < 0.05
        n_positive = len(positive_methods)
        tost_equiv = tost.get("equivalent", False)

        # -- Priority cascade ---------------------------------------------

        # 1. MANDATORY -- resample/ablation proves causal necessity
        if causal:
            return self._make(
                "MANDATORY",
                confidence=0.99,
                evidence=evidence_bundle,
            )

        # 2. SUPERPOSED_NON_ZOMBIE -- SAE detects superposition + ridge low
        if sae_detected and ridge_low:
            return self._make(
                "SUPERPOSED_NON_ZOMBIE",
                confidence=0.90,
                evidence=evidence_bundle,
            )

        # 3. NONLINEAR_ENCODED -- MLP high + ridge low (nonlinear encoding)
        if mlp_high and ridge_low:
            return self._make(
                "NONLINEAR_ENCODED",
                confidence=0.85,
                evidence=evidence_bundle,
            )

        # 4. SPURIOUS_DRIFT -- frequency drift only (no other positive signal)
        if freq_detected and not sig and n_positive <= 1:
            return self._make(
                "SPURIOUS_DRIFT",
                confidence=0.80,
                evidence=evidence_bundle,
            )

        # 5. CONFIRMED_NON_ZOMBIE -- significant + multiple methods agree
        if sig and n_positive >= 3:
            return self._make(
                "CONFIRMED_NON_ZOMBIE",
                confidence=0.88,
                evidence=evidence_bundle,
            )

        # 6. CANDIDATE_ENCODED -- some positive signal but not enough
        if sig and n_positive >= 1:
            return self._make(
                "CANDIDATE_ENCODED",
                confidence=0.60,
                evidence=evidence_bundle,
            )

        # 7. CONFIRMED_ZOMBIE -- TOST equivalence + strong BF for null
        if tost_equiv and bf01 > 3.0:
            return self._make(
                "CONFIRMED_ZOMBIE",
                confidence=0.92,
                evidence=evidence_bundle,
            )

        # 8. LIKELY_ZOMBIE -- TOST equivalence alone
        if tost_equiv:
            return self._make(
                "LIKELY_ZOMBIE",
                confidence=0.70,
                evidence=evidence_bundle,
            )

        # 9. Fallback -- AMBIGUOUS
        return self._make(
            "AMBIGUOUS",
            confidence=0.30,
            evidence=evidence_bundle,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make(verdict: str, confidence: float, evidence: dict) -> dict:
        """Build a standardised verdict dictionary."""
        return {
            "verdict": verdict,
            "confidence": confidence,
            "evidence": evidence,
        }
