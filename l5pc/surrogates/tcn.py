"""
L5PC DESCARTES -- Temporal Convolutional Network Surrogate

Replicates the Beniaguev et al. (2022) architecture:
    7 layers, 128 features, causal dilated convolutions.
    Dilations: 1, 2, 4, 8, 16, 32, 64
    Receptive field: sum(2*(128-1)*d for d in dilations)*dt = 153 ms at dt=0.5ms

Supports:
    - Training from scratch on Bahl reduced-model inputs (50 channels)
    - Loading pre-trained Beniaguev weights from TensorFlow/Keras .h5
      (requires full 639-compartment Hay model input format)
    - Hidden-state extraction from all intermediate layers for probing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from l5pc.config import (
    TOTAL_SYN,
    TCN_N_LAYERS,
    TCN_N_FEATURES,
    TCN_PRETRAINED_PATH,
    HAY_N_COMPARTMENTS,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """1-D convolution with left-padding to preserve causality.

    For a kernel of size *k* and dilation *d*, the receptive field extends
    ``(k-1)*d`` steps into the past.  We pad that many zeros on the left
    and zero on the right so the output length equals the input length and
    no future information leaks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,  # we handle padding manually
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, T)"""
        x = F.pad(x, (self.padding, 0))  # left-pad only
        return self.conv(x)


class TCNBlock(nn.Module):
    """Residual TCN block.

    Structure per block:
        CausalConv1d -> ReLU -> Dropout ->
        CausalConv1d -> ReLU -> Dropout ->
        + residual (with 1x1 conv if channel dims differ)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        # 1x1 projection for residual when channel counts change
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, T)"""
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        res = self.residual(x)
        return self.final_relu(out + res)


# ---------------------------------------------------------------------------
# Full TCN model
# ---------------------------------------------------------------------------

class L5PC_TCN(nn.Module):
    """Temporal Convolutional Network for L5PC spike prediction.

    Parameters
    ----------
    input_dim : int
        Number of input channels per timestep.
    n_layers : int
        Number of TCN blocks (default: 7).
    n_features : int
        Hidden feature channels in every TCN block (default: 128).
    kernel_size : int
        Convolution kernel width (default: 2, matching Beniaguev).
    dropout : float
        Dropout probability inside TCN blocks.
    """

    def __init__(
        self,
        input_dim: int = TOTAL_SYN,
        n_layers: int = TCN_N_LAYERS,
        n_features: int = TCN_N_FEATURES,
        kernel_size: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.n_features = n_features

        blocks = []
        for i in range(n_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, 16, 32, 64
            in_ch = input_dim if i == 0 else n_features
            blocks.append(
                TCNBlock(in_ch, n_features, kernel_size, dilation, dropout)
            )
        self.blocks = nn.ModuleList(blocks)

        # Pointwise readout: n_features -> 1
        self.readout = nn.Conv1d(n_features, 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, T, input_dim).  Internally transposed to
            (batch, input_dim, T) for Conv1d operations.
        return_hidden : bool
            If True, also return concatenated hidden activations from every
            block for downstream probing.

        Returns
        -------
        output : torch.Tensor
            Shape (batch, T) -- spike probability in [0, 1].
        hidden_states : torch.Tensor, optional
            Shape (batch, T, n_layers * n_features) -- returned only when
            ``return_hidden=True``.
        """
        # (batch, T, C) -> (batch, C, T)
        h = x.transpose(1, 2)

        hiddens = []
        for block in self.blocks:
            h = block(h)
            if return_hidden:
                # h: (batch, n_features, T) -> store transposed
                hiddens.append(h.transpose(1, 2))

        # Readout
        logits = self.readout(h).squeeze(1)  # (batch, T)
        output = torch.sigmoid(logits)

        if return_hidden:
            # Concatenate along feature dim: (batch, T, n_layers * n_features)
            hidden_states = torch.cat(hiddens, dim=-1)
            return output, hidden_states
        return (output,)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"L5PC_TCN(input_dim={self.input_dim}, "
            f"n_layers={self.n_layers}, "
            f"n_features={self.n_features}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# Pre-trained weight loading (Beniaguev TF/Keras -> PyTorch)
# ---------------------------------------------------------------------------

def load_pretrained_tcn(
    model_path: str = None,
    device: str = "cpu",
) -> L5PC_TCN:
    """Load Beniaguev pre-trained TCN from a TensorFlow/Keras ``.h5`` file.

    The Beniaguev model expects the full 639-compartment Hay model input.
    Weights are extracted layer-by-layer from the Keras model and mapped
    into the equivalent PyTorch ``L5PC_TCN`` architecture.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the ``.h5`` weights file.  Defaults to
        ``config.TCN_PRETRAINED_PATH``.
    device : str
        Target torch device.

    Returns
    -------
    model : L5PC_TCN
        PyTorch TCN with Beniaguev weights loaded.  ``input_dim`` is set
        to ``HAY_N_COMPARTMENTS`` (639).

    Notes
    -----
    Requires ``tensorflow`` to be installed.  Only works with the full
    639-compartment format; the Bahl reduced model (50 inputs) uses
    separately trained weights.
    """
    import tensorflow as tf

    if model_path is None:
        model_path = str(TCN_PRETRAINED_PATH)

    # Load Keras model
    tf_model = tf.keras.models.load_model(model_path, compile=False)

    # Build PyTorch model with matching input dim
    pt_model = L5PC_TCN(
        input_dim=HAY_N_COMPARTMENTS,
        n_layers=TCN_N_LAYERS,
        n_features=TCN_N_FEATURES,
    )

    # Map Keras conv layers to PyTorch TCN blocks
    tf_conv_layers = [
        layer for layer in tf_model.layers
        if "conv1d" in layer.name.lower()
    ]

    conv_idx = 0
    for block in pt_model.blocks:
        for pt_conv in [block.conv1, block.conv2]:
            if conv_idx >= len(tf_conv_layers):
                break
            tf_layer = tf_conv_layers[conv_idx]
            weights = tf_layer.get_weights()
            # Keras Conv1D weight shape: (kernel_size, in_channels, out_channels)
            # PyTorch Conv1d weight shape: (out_channels, in_channels, kernel_size)
            kernel = np.array(weights[0])
            kernel_pt = np.transpose(kernel, (2, 1, 0))
            pt_conv.conv.weight.data = torch.tensor(
                kernel_pt, dtype=torch.float32
            )
            if len(weights) > 1:
                pt_conv.conv.bias.data = torch.tensor(
                    np.array(weights[1]), dtype=torch.float32
                )
            conv_idx += 1

    # Handle the final readout / dense layer
    tf_dense_layers = [
        layer for layer in tf_model.layers
        if "dense" in layer.name.lower()
    ]
    if tf_dense_layers:
        dense = tf_dense_layers[-1]
        d_weights = dense.get_weights()
        # Dense weight shape: (in_features, out_features) ->
        # Conv1d(1x1) shape: (out_features, in_features, 1)
        w = np.array(d_weights[0])  # (n_features, 1)
        pt_model.readout.weight.data = torch.tensor(
            w.T[:, :, np.newaxis], dtype=torch.float32
        )
        if len(d_weights) > 1:
            pt_model.readout.bias.data = torch.tensor(
                np.array(d_weights[1]), dtype=torch.float32
            )

    pt_model.to(device)
    pt_model.eval()
    return pt_model


# ---------------------------------------------------------------------------
# Hidden-state extraction utility
# ---------------------------------------------------------------------------

def extract_tcn_hidden(
    model: L5PC_TCN,
    x: torch.Tensor,
) -> torch.Tensor:
    """Extract hidden activations from all TCN layers.

    Parameters
    ----------
    model : L5PC_TCN
        Trained (or untrained) TCN model.
    x : torch.Tensor
        Input tensor, shape (batch, T, input_dim).

    Returns
    -------
    hidden : torch.Tensor
        Shape (batch, T, n_layers * n_features).
        With default settings: (batch, T, 7 * 128) = (batch, T, 896).
    """
    model.eval()
    with torch.no_grad():
        _, hidden = model(x, return_hidden=True)
    return hidden
