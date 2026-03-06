"""
L5PC DESCARTES -- LSTM Surrogate for Voltage Prediction

Architecture matches the hippocampal DESCARTES experiment to enable
cross-circuit comparison of learned representations.

Input:  (batch, T, input_dim)  -- synaptic current traces per timestep
Output: (batch, T)             -- predicted somatic voltage at each timestep

The output is unbounded (no sigmoid) because the training target is raw
somatic membrane voltage in millivolts (range ~[-80, +40] mV).

Hidden states (batch, T, hidden_size) are the primary substrate for
linear probing in the DESCARTES pipeline.
"""

import torch
import torch.nn as nn

from l5pc.config import (
    N_LSTM_LAYERS,
    TOTAL_SYN,
)


class L5PC_LSTM(nn.Module):
    """LSTM surrogate mapping synaptic inputs to somatic membrane voltage.

    Parameters
    ----------
    input_dim : int
        Number of input channels per timestep (default: TOTAL_SYN = 50).
    hidden_size : int
        LSTM hidden dimension.  Swept over HIDDEN_SIZES = [64, 128, 256].
    n_layers : int
        Number of stacked LSTM layers (default: N_LSTM_LAYERS = 2).
    dropout : float
        Dropout between LSTM layers (applied only when n_layers > 1).
    """

    def __init__(
        self,
        input_dim: int = TOTAL_SYN,
        hidden_size: int = 128,
        n_layers: int = N_LSTM_LAYERS,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Linear readout: hidden_size -> 1 scalar per timestep
        self.readout = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, T, input_dim).
        return_hidden : bool
            If True, also return the full sequence of hidden states from the
            last LSTM layer for downstream probing.

        Returns
        -------
        output : torch.Tensor
            Shape (batch, T) -- predicted somatic voltage (mV, unbounded).
        hidden_states : torch.Tensor, optional
            Shape (batch, T, hidden_size) -- returned only when
            ``return_hidden=True``.
        """
        # lstm_out: (batch, T, hidden_size) -- last-layer hidden states
        lstm_out, _ = self.lstm(x)

        # readout: (batch, T, 1) -> squeeze -> (batch, T)
        # No sigmoid -- output is raw voltage in millivolts
        output = self.readout(lstm_out).squeeze(-1)

        if return_hidden:
            return output, lstm_out
        return (output,)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"L5PC_LSTM(input_dim={self.input_dim}, "
            f"hidden_size={self.hidden_size}, "
            f"n_layers={self.n_layers}, "
            f"params={self.count_parameters():,})"
        )
