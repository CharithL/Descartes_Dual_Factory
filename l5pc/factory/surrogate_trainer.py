"""Surrogate model trainer for the DESCARTES Dual Factory v3.0.

Handles model construction, training with early stopping,
regularization, and output validation via cross-condition correlation.
"""

import numpy as np
import torch
import torch.nn as nn

from .surrogate_genome import SurrogateGenome_v3

# ---------------------------------------------------------------------------
# Output gate threshold -- surrogate must exceed this CC to proceed to probing
# ---------------------------------------------------------------------------
OUTPUT_CC_THRESHOLD = 0.7


class SurrogateTrainer:
    """Train and validate a surrogate model specified by a SurrogateGenome_v3."""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train_and_validate(
        self,
        genome: SurrogateGenome_v3,
        train_data: dict,
        val_data: dict,
        device: torch.device,
    ) -> dict:
        """Build, train, and gate-check a surrogate model.

        Parameters
        ----------
        genome : SurrogateGenome_v3
            Full architecture/training specification.
        train_data : dict
            Training split with keys ``'X'`` and ``'y'`` (numpy or tensor).
        val_data : dict
            Validation split with keys ``'X'`` and ``'y'``.
        device : torch.device
            Target compute device (cpu / cuda).

        Returns
        -------
        dict
            Keys: model, passed_gate, output_cc, history, genome.
        """
        model = self._build_model(genome, train_data)
        model = model.to(device)

        history = self._train(model, genome, train_data, val_data, device)

        output_cc = self._cross_condition_correlation(model, val_data, device)
        passed_gate = output_cc >= OUTPUT_CC_THRESHOLD

        return {
            "model": model,
            "passed_gate": passed_gate,
            "output_cc": output_cc,
            "history": history,
            "genome": genome,
        }

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(
        self,
        genome: SurrogateGenome_v3,
        data: dict,
    ) -> nn.Module:
        """Dispatch to the appropriate architecture builder.

        Architecture classes are imported on demand so that unused backends
        do not introduce hard dependencies.

        Raises
        ------
        ValueError
            If ``genome.architecture`` is not recognised.
        """
        arch = genome.architecture

        if arch == "lstm":
            from ..architectures.lstm import LSTMSurrogate
            return LSTMSurrogate(genome, data)
        elif arch == "gru":
            from ..architectures.gru import GRUSurrogate
            return GRUSurrogate(genome, data)
        elif arch == "neural_ode":
            from ..architectures.neural_ode import NeuralODESurrogate
            return NeuralODESurrogate(genome, data)
        elif arch == "ude":
            from ..architectures.ude import UDESurrogate
            return UDESurrogate(genome, data)
        elif arch == "ltc":
            from ..architectures.ltc import LTCSurrogate
            return LTCSurrogate(genome, data)
        elif arch == "rmm":
            from ..architectures.rmm import RMMSurrogate
            return RMMSurrogate(genome, data)
        elif arch == "mamba":
            from ..architectures.mamba import MambaSurrogate
            return MambaSurrogate(genome, data)
        elif arch == "neural_cde":
            from ..architectures.neural_cde import NeuralCDESurrogate
            return NeuralCDESurrogate(genome, data)
        elif arch == "koopman_ae":
            from ..architectures.koopman_ae import KoopmanAESurrogate
            return KoopmanAESurrogate(genome, data)
        elif arch == "volterra":
            from ..architectures.volterra import VolterraSurrogate
            return VolterraSurrogate(genome, data)
        elif arch == "tcn":
            from ..architectures.tcn import TCNSurrogate
            return TCNSurrogate(genome, data)
        elif arch == "transformer":
            from ..architectures.transformer import TransformerSurrogate
            return TransformerSurrogate(genome, data)
        elif arch == "pinn":
            from ..architectures.pinn import PINNSurrogate
            return PINNSurrogate(genome, data)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    # ------------------------------------------------------------------
    # Training loop (with early stopping)
    # ------------------------------------------------------------------

    def _train(
        self,
        model: nn.Module,
        genome: SurrogateGenome_v3,
        train_data: dict,
        val_data: dict,
        device: torch.device,
    ) -> dict:
        """Full training loop with early stopping.

        Returns
        -------
        dict
            History with keys ``'train_loss'``, ``'val_loss'``, ``'val_cc'``.
        """
        loss_fn = self._get_loss_fn(genome.loss)
        optimizer = self._build_optimizer(model, genome)

        epochs = getattr(genome, "epochs", 200)
        patience = getattr(genome, "patience", 20)

        history: dict = {"train_loss": [], "val_loss": [], "val_cc": []}
        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(epochs):
            train_loss = self._train_epoch(
                model, train_data, loss_fn, optimizer, genome, device
            )
            val_loss, val_cc = self._validate(model, val_data, loss_fn, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_cc"].append(val_cc)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        # Restore best checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)

        return history

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------

    def _compute_regularization(
        self,
        model: nn.Module,
        genome: SurrogateGenome_v3,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute composite regularisation penalty.

        Components
        ----------
        * **L1 sparsity** on hidden activations (``hidden_l1``).
        * **Slow-feature penalty** penalising temporal derivatives of
          hidden states (``slow_feature_penalty``).
        * **Information-bottleneck** (disentanglement) penalty via variance
          of hidden activations (``disentanglement_penalty``).
        """
        reg = torch.tensor(0.0, device=hidden_states.device)

        # L1 sparsity on hidden activations
        l1_weight = getattr(genome, "hidden_l1", 0.0)
        if l1_weight > 0.0:
            reg = reg + l1_weight * hidden_states.abs().mean()

        # Slow-feature penalty: penalise large temporal differences
        slow_weight = getattr(genome, "slow_feature_penalty", 0.0)
        if slow_weight > 0.0 and hidden_states.dim() >= 2 and hidden_states.size(0) > 1:
            diffs = hidden_states[1:] - hidden_states[:-1]
            reg = reg + slow_weight * diffs.pow(2).mean()

        # Information-bottleneck / disentanglement penalty
        ib_weight = getattr(genome, "disentanglement_penalty", 0.0)
        if ib_weight > 0.0:
            var = hidden_states.var(dim=0).mean()
            reg = reg + ib_weight * var

        return reg

    # ------------------------------------------------------------------
    # Loss function dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _get_loss_fn(loss_name: str) -> nn.Module:
        """Return the PyTorch loss module for *loss_name*.

        Supported: ``'mse'``, ``'bce'``, ``'poisson_nll'``, ``'huber'``.

        Raises
        ------
        ValueError
            If *loss_name* is not recognised.
        """
        loss_name = loss_name.lower()
        if loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "bce":
            return nn.BCEWithLogitsLoss()
        elif loss_name == "poisson_nll":
            return nn.PoissonNLLLoss(log_input=True)
        elif loss_name == "huber":
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    # ------------------------------------------------------------------
    # Optimizer dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _build_optimizer(
        model: nn.Module,
        genome: SurrogateGenome_v3,
    ) -> torch.optim.Optimizer:
        """Build the optimizer specified in the genome.

        Supported: ``'adam'``, ``'adamw'``, ``'sgd'``.

        Raises
        ------
        ValueError
            If ``genome.optimizer`` is not recognised.
        """
        lr = getattr(genome, "lr", 1e-3)
        wd = getattr(genome, "weight_decay", 0.0)
        opt_name = getattr(genome, "optimizer", "adam").lower()

        if opt_name == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            momentum = getattr(genome, "momentum", 0.9)
            return torch.optim.SGD(
                model.parameters(), lr=lr, weight_decay=wd, momentum=momentum
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    # ------------------------------------------------------------------
    # Cross-condition correlation (output gate metric)
    # ------------------------------------------------------------------

    def _cross_condition_correlation(
        self,
        model: nn.Module,
        val_data: dict,
        device: torch.device,
    ) -> float:
        """Pearson correlation between model predictions and targets.

        Operates across all validation conditions so the metric captures
        generalisation rather than in-condition fit alone.

        Returns
        -------
        float
            Pearson correlation coefficient (clipped to [0, 1]).
        """
        model.eval()  # noqa: B010
        with torch.no_grad():
            X = val_data["X"]
            y = val_data["y"]
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y).float()
            X = X.to(device)
            y = y.to(device)

            preds = model(X)
            if isinstance(preds, tuple):
                preds = preds[0]  # Some architectures return (output, hidden)

            preds_flat = preds.reshape(-1).cpu().numpy()
            targets_flat = y.reshape(-1).cpu().numpy()

            if preds_flat.std() < 1e-8 or targets_flat.std() < 1e-8:
                return 0.0

            cc = float(np.corrcoef(preds_flat, targets_flat)[0, 1])
            return max(0.0, cc)

    # ------------------------------------------------------------------
    # Single training epoch
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: nn.Module,
        data: dict,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        genome: SurrogateGenome_v3,
        device: torch.device,
    ) -> float:
        """Run one training epoch and return the mean loss."""
        model.train()

        X = data["X"]
        y = data["y"]
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(X)
        hidden_states = None
        if isinstance(output, tuple):
            output, hidden_states = output

        loss = loss_fn(output, y)

        # Regularization (if hidden states available)
        if hidden_states is not None:
            reg = self._compute_regularization(model, genome, hidden_states)
            loss = loss + reg

        loss.backward()
        optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(
        self,
        model: nn.Module,
        data: dict,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> tuple:
        """Run on validation data.

        Returns
        -------
        tuple[float, float]
            ``(val_loss, val_cc)`` where *val_cc* is the Pearson
            correlation between predictions and targets.
        """
        model.eval()  # noqa: B010
        with torch.no_grad():
            X = data["X"]
            y = data["y"]
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y).float()
            X = X.to(device)
            y = y.to(device)

            output = model(X)
            if isinstance(output, tuple):
                output = output[0]

            val_loss = float(loss_fn(output, y).item())

            preds_flat = output.reshape(-1).cpu().numpy()
            targets_flat = y.reshape(-1).cpu().numpy()

            if preds_flat.std() < 1e-8 or targets_flat.std() < 1e-8:
                val_cc = 0.0
            else:
                val_cc = float(np.corrcoef(preds_flat, targets_flat)[0, 1])
                val_cc = max(0.0, val_cc)

        return val_loss, val_cc
