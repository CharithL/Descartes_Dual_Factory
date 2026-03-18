"""
DESCARTES Dual Factory v3.0 -- C1 Inner Loop: Probing Factory Evaluator

The ProbingFactoryEvaluator runs the tiered probing pipeline on a trained
surrogate model to determine whether its hidden states encode biological
variables.  This is the inner loop of the dual factory: every candidate
surrogate that survives output-accuracy gating is subjected to this
evaluation before a fitness score is assigned.

Tier 0: Ridge + MLP hardened probe (per-variable).  Early termination
         if ALL variables are zombie.
Tier 1: SAE decomposition + CCA + RSA (joint alignment / superposition).
         Early termination if no evidence of biological encoding.
Tier 2: Resample ablation on candidates that passed earlier tiers.
         Counts mandatory variables.
"""

import copy
import logging
from collections import defaultdict

import numpy as np

# Placeholder imports for probing functions.
# These will be available when the full pipeline is installed.
try:
    from ..probing.hardening import hardened_probe
    from ..probing.sae_probe import train_sae, sae_probe_biological_variables
    from ..probing.joint_alignment import cca_alignment, rsa_comparison
    from ..probing.ablation import resample_ablation
except ImportError:
    pass  # Will be available when full pipeline is installed

logger = logging.getLogger(__name__)


# ── Verdict constants ──────────────────────────────────────────────────
ZOMBIE_VERDICTS = {'CONFIRMED_ZOMBIE', 'LIKELY_ZOMBIE'}
ENCODED_VERDICTS = {
    'CONFIRMED_ENCODED', 'CANDIDATE_ENCODED', 'NONLINEAR_ENCODED',
}
SUPERPOSED_THRESHOLD = 0.05   # SAE R2 gain over raw to declare superposition
CCA_SIGNIFICANCE = 0.05       # p-value threshold for CCA
RSA_SIGNIFICANCE = 0.05       # p-value threshold for RSA
CAUSAL_MANDATORY_Z = -2.0     # z-score below which variable is mandatory


class ProbingFactoryEvaluator:
    """C1 inner loop: tiered probing evaluation of a surrogate model.

    Parameters
    ----------
    bio_targets : dict[str, np.ndarray]
        Mapping from variable name to (N_samples,) target array.
    target_names : list[str]
        Ordered list of biological variable names.
    full_evaluation : bool
        If True, run all tiers regardless of early termination.
        If False (default), apply early termination at each tier gate.
    """

    def __init__(self, bio_targets, target_names, full_evaluation=False):
        self.bio_targets = bio_targets
        self.target_names = list(target_names)
        self.full_evaluation = full_evaluation

    # ── Public API ─────────────────────────────────────────────────────

    def evaluate_surrogate(self, model, circuit_data, device='cpu'):
        """Run the tiered probing pipeline on a trained surrogate.

        Parameters
        ----------
        model : nn.Module
            Trained surrogate model (must support forward pass).
        circuit_data : dict
            Circuit data with 'test_inputs' and 'test_outputs' keys.
        device : str
            Torch device string.

        Returns
        -------
        dict
            verdict : str
                Final verdict string from _generate_final_verdict.
            tier_reached : int
                Highest tier completed (0, 1, or 2).
            n_mandatory_variables : int
                Number of variables confirmed as causally mandatory.
            per_variable : dict
                Per-variable results from all completed tiers.
            joint_alignment : dict or None
                CCA/RSA results from Tier 1, or None if not reached.
            causal : dict or None
                Resample ablation results from Tier 2, or None if not reached.
        """
        # Extract hidden states
        h_trained = self._extract_hidden(model, circuit_data, device)
        h_untrained = self._extract_hidden_untrained(model, circuit_data, device)

        per_variable = {}
        joint_alignment = None
        causal = None
        tier_reached = 0

        # ── Tier 0: Hardened probe per variable ────────────────────────
        logger.info("Tier 0: Running hardened probes for %d variables",
                    len(self.target_names))

        tier0_results = {}
        for name in self.target_names:
            target = self.bio_targets[name]
            result = hardened_probe(
                h_trained, h_untrained, target, name, device=device)
            tier0_results[name] = result
            per_variable[name] = {
                'tier0': result,
                'hardened_verdict': result['hardened_verdict'],
            }

        # Early termination: if ALL variables are zombie, stop
        all_zombie = all(
            tier0_results[n]['hardened_verdict'] in ZOMBIE_VERDICTS
            for n in self.target_names
        )
        if all_zombie and not self.full_evaluation:
            logger.info("Tier 0 early termination: all variables are zombie")
            n_mandatory = 0
            verdict = self._generate_final_verdict(
                tier0_results, None, None, n_mandatory)
            return {
                'verdict': verdict,
                'tier_reached': 0,
                'n_mandatory_variables': n_mandatory,
                'per_variable': per_variable,
                'joint_alignment': None,
                'causal': None,
            }

        # Identify candidates that survived Tier 0
        tier0_survivors = [
            n for n in self.target_names
            if tier0_results[n]['hardened_verdict'] not in ZOMBIE_VERDICTS
        ]
        logger.info("Tier 0 survivors: %d / %d",
                    len(tier0_survivors), len(self.target_names))

        # ── Tier 1: SAE + CCA + RSA ───────────────────────────────────
        tier_reached = 1
        logger.info("Tier 1: SAE decomposition + joint alignment")

        # Train SAE on hidden states
        hidden_dim = h_trained.shape[-1]
        hidden_list = [h_trained]  # train_sae expects list of arrays
        sae, sae_loss = train_sae(hidden_list, hidden_dim, device=device)

        # SAE probe for biological variables
        bio_target_matrix = np.column_stack(
            [self.bio_targets[n] for n in self.target_names])
        sae_results = sae_probe_biological_variables(
            sae, hidden_list, bio_target_matrix, self.target_names,
            device=device)

        # Check for superposition evidence
        for name in self.target_names:
            if name in sae_results:
                per_variable[name]['tier1_sae'] = sae_results[name]
                raw_r2 = tier0_results[name].get('ridge_delta_r2', 0.0)
                sae_r2 = sae_results[name].get('sae_r2', 0.0)
                per_variable[name]['superposition'] = (
                    sae_r2 - raw_r2 > SUPERPOSED_THRESHOLD)

        # CCA alignment
        cca_result = cca_alignment(h_trained, bio_target_matrix)

        # RSA comparison
        rsa_result = rsa_comparison(h_trained, bio_target_matrix)

        joint_alignment = {
            'cca': cca_result,
            'rsa': rsa_result,
        }

        # Early termination: no evidence of biological encoding
        has_cca_evidence = (
            cca_result.get('p_value', 1.0) < CCA_SIGNIFICANCE)
        has_rsa_evidence = (
            rsa_result.get('p_value', 1.0) < RSA_SIGNIFICANCE)
        has_sae_evidence = any(
            per_variable[n].get('superposition', False)
            for n in self.target_names
        )
        has_any_encoded = len(tier0_survivors) > 0

        no_evidence = (
            not has_cca_evidence
            and not has_rsa_evidence
            and not has_sae_evidence
            and not has_any_encoded
        )
        if no_evidence and not self.full_evaluation:
            logger.info("Tier 1 early termination: no encoding evidence")
            n_mandatory = 0
            verdict = self._generate_final_verdict(
                tier0_results, joint_alignment, None, n_mandatory)
            return {
                'verdict': verdict,
                'tier_reached': 1,
                'n_mandatory_variables': n_mandatory,
                'per_variable': per_variable,
                'joint_alignment': joint_alignment,
                'causal': None,
            }

        # ── Tier 2: Resample ablation on surviving candidates ──────────
        tier_reached = 2
        logger.info("Tier 2: Resample ablation on %d candidates",
                    len(tier0_survivors))

        test_inputs = circuit_data['test_inputs']
        test_outputs = circuit_data['test_outputs']
        hidden_states = h_trained

        causal = {}
        n_mandatory = 0

        for name in tier0_survivors:
            target = self.bio_targets[name]
            ablation_result = resample_ablation(
                model, test_inputs, test_outputs,
                target, hidden_states)
            causal[name] = ablation_result
            per_variable[name]['tier2_causal'] = ablation_result

            # Check if mandatory
            z_scores = ablation_result.get('z_scores', [])
            if len(z_scores) > 0 and np.min(z_scores) < CAUSAL_MANDATORY_Z:
                per_variable[name]['mandatory'] = True
                n_mandatory += 1
            else:
                per_variable[name]['mandatory'] = False

        logger.info("Tier 2 complete: %d mandatory variables", n_mandatory)

        # ── Generate final verdict ─────────────────────────────────────
        verdict = self._generate_final_verdict(
            tier0_results, joint_alignment, causal, n_mandatory)

        return {
            'verdict': verdict,
            'tier_reached': tier_reached,
            'n_mandatory_variables': n_mandatory,
            'per_variable': per_variable,
            'joint_alignment': joint_alignment,
            'causal': causal,
        }

    # ── Verdict generation ─────────────────────────────────────────────

    def _generate_final_verdict(self, tier0, tier1, tier2, n_mandatory):
        """Generate a final verdict string from all tier results.

        Parameters
        ----------
        tier0 : dict
            Per-variable hardened probe results.
        tier1 : dict or None
            Joint alignment results (CCA/RSA).
        tier2 : dict or None
            Causal ablation results.
        n_mandatory : int
            Number of causally mandatory variables.

        Returns
        -------
        str
            Final verdict string.
        """
        if tier0 is None:
            return 'NO_DATA'

        # Count encoding verdicts from Tier 0
        n_encoded = sum(
            1 for name in self.target_names
            if tier0[name]['hardened_verdict'] in ENCODED_VERDICTS
        )
        n_zombie = sum(
            1 for name in self.target_names
            if tier0[name]['hardened_verdict'] in ZOMBIE_VERDICTS
        )
        n_total = len(self.target_names)

        # All zombie
        if n_encoded == 0:
            return 'ALL_ZOMBIE'

        # No causal tier reached
        if tier2 is None:
            if n_encoded > 0:
                return 'ENCODED_NO_CAUSAL'
            return 'INCONCLUSIVE'

        # Causal results available
        if n_mandatory == 0:
            return 'ENCODED_NOT_CAUSAL'

        if n_mandatory >= n_total * 0.5:
            return 'STRONG_BIOLOGICAL_SURROGATE'

        if n_mandatory >= 1:
            return 'PARTIAL_BIOLOGICAL_SURROGATE'

        return 'INCONCLUSIVE'

    # ── Hidden state extraction ────────────────────────────────────────

    def _extract_hidden(self, model, circuit_data, device='cpu'):
        """Extract hidden states from a trained model.

        Parameters
        ----------
        model : nn.Module
            Trained surrogate model.
        circuit_data : dict
            Must contain 'test_inputs'.
        device : str
            Torch device.

        Returns
        -------
        np.ndarray
            Hidden states array of shape (N_samples, hidden_dim).
        """
        import torch

        model.eval()
        model.to(device)
        test_inputs = circuit_data['test_inputs']

        if isinstance(test_inputs, np.ndarray):
            test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
        test_inputs = test_inputs.to(device)

        with torch.no_grad():
            # Model should return (output, hidden_states) or have
            # a get_hidden() method
            if hasattr(model, 'get_hidden'):
                hidden = model.get_hidden(test_inputs)
            else:
                output = model(test_inputs)
                if isinstance(output, tuple) and len(output) >= 2:
                    hidden = output[1]
                else:
                    raise ValueError(
                        "Model must return (output, hidden) tuple or "
                        "implement get_hidden()")

        if isinstance(hidden, torch.Tensor):
            hidden = hidden.cpu().numpy()

        # Flatten temporal dimension if 3D: (batch, time, hidden) -> (N, hidden)
        if hidden.ndim == 3:
            hidden = hidden.reshape(-1, hidden.shape[-1])

        return hidden

    def _extract_hidden_untrained(self, model, circuit_data, device='cpu'):
        """Extract hidden states from an untrained (re-initialised) copy.

        Creates a deep copy of the model, re-initialises all parameters,
        and runs the same extraction pipeline.

        Parameters
        ----------
        model : nn.Module
            The trained model (will NOT be modified).
        circuit_data : dict
            Must contain 'test_inputs'.
        device : str
            Torch device.

        Returns
        -------
        np.ndarray
            Hidden states from the untrained model.
        """
        import torch

        # Deep copy and reinitialise
        untrained = copy.deepcopy(model)
        for name, param in untrained.named_parameters():
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)

        return self._extract_hidden(untrained, circuit_data, device)
