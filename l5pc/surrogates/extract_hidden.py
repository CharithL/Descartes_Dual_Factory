"""
L5PC DESCARTES -- Hidden State Extraction

Extract hidden representations from trained AND untrained (random-init)
surrogate models on the test set.  The untrained baseline is critical:
the primary DESCARTES metric is

    delta_R2 = R2_trained - R2_untrained

which isolates learned representations from random-projection artefacts.

Output format: ``.npz`` files with key ``hidden_states`` containing a
numpy array of shape ``(total_T, hidden_dim)`` where
``total_T = n_test_trials * T_STEPS``.

Usage
-----
    from l5pc.surrogates.extract_hidden import extract_all
    extract_all(data_dir='data/bahl_trials', surrogate_dir='data/surrogates')
"""

import logging
from pathlib import Path
from typing import Optional, Type

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from l5pc.config import (
    BATCH_SIZE,
    BAHL_TRIAL_DIR,
    HIDDEN_SIZES,
    N_LSTM_LAYERS,
    SURROGATE_DIR,
    TCN_N_FEATURES,
    TCN_N_LAYERS,
    TOTAL_SYN,
)
from l5pc.surrogates.lstm import L5PC_LSTM
from l5pc.surrogates.tcn import L5PC_TCN
from l5pc.surrogates.train import create_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_hidden_states(
    model: nn.Module,
    test_data: torch.Tensor,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Run model on test data and collect hidden states.

    Parameters
    ----------
    model : nn.Module
        Must support ``forward(x, return_hidden=True)`` returning
        ``(output, hidden_states)`` where hidden_states has shape
        ``(batch, T, hidden_dim)``.
    test_data : torch.Tensor
        Input tensor of shape ``(n_trials, T, input_dim)``.
    batch_size : int
        Inference batch size.

    Returns
    -------
    hidden : np.ndarray
        Shape ``(n_trials * T, hidden_dim)`` -- all timesteps flattened
        across trials.
    """
    device = next(model.parameters()).device
    model.eval()

    all_hidden = []
    n_trials = test_data.size(0)

    with torch.no_grad():
        for start in range(0, n_trials, batch_size):
            end = min(start + batch_size, n_trials)
            x_batch = test_data[start:end].to(device)

            _, hidden = model(x_batch, return_hidden=True)
            # hidden: (batch, T, hidden_dim) -> reshape to (batch*T, hidden_dim)
            b, t, d = hidden.shape
            all_hidden.append(hidden.reshape(b * t, d).cpu().numpy())

    hidden = np.concatenate(all_hidden, axis=0)
    logger.info(
        "Extracted hidden states: shape %s (%.1f MB)",
        hidden.shape, hidden.nbytes / 1e6,
    )
    return hidden


# ---------------------------------------------------------------------------
# Extract trained + untrained pair
# ---------------------------------------------------------------------------

def extract_and_save(
    model_class: Type[nn.Module],
    model_path: str,
    test_data_dir: str,
    trained_path: str,
    untrained_path: str,
    **model_kwargs,
) -> None:
    """Extract hidden states from a trained model and its untrained baseline.

    Steps:
        1. Instantiate model with ``model_kwargs``, load trained weights
           from ``model_path``, run on test set, save as ``.npz``.
        2. Instantiate a fresh model with identical architecture (random
           init), run on test set, save as ``.npz``.

    Parameters
    ----------
    model_class : type
        ``L5PC_LSTM`` or ``L5PC_TCN``.
    model_path : str or Path
        Path to the trained model checkpoint (``.pt`` file).
    test_data_dir : str or Path
        Directory containing trial data (``inputs.npy``, ``outputs.npy``).
    trained_path : str or Path
        Output path for trained hidden states (``.npz``).
    untrained_path : str or Path
        Output path for untrained hidden states (``.npz``).
    **model_kwargs
        Keyword arguments forwarded to ``model_class()``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_ds = create_dataset(test_data_dir, "test")
    test_inputs = test_ds.tensors[0]  # (n_test, T, input_dim)

    trained_path = Path(trained_path)
    untrained_path = Path(untrained_path)
    trained_path.parent.mkdir(parents=True, exist_ok=True)
    untrained_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Trained model ---
    logger.info("Extracting TRAINED hidden states from %s", model_path)
    trained_model = model_class(**model_kwargs).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    trained_model.load_state_dict(state_dict)

    trained_hidden = extract_hidden_states(trained_model, test_inputs)
    np.savez_compressed(trained_path, hidden_states=trained_hidden)
    logger.info("Saved trained hidden states to %s", trained_path)

    # --- Untrained (random init) baseline ---
    logger.info("Extracting UNTRAINED hidden states (random init)")
    untrained_model = model_class(**model_kwargs).to(device)
    # No weight loading -- random initialization

    untrained_hidden = extract_hidden_states(untrained_model, test_inputs)
    np.savez_compressed(untrained_path, hidden_states=untrained_hidden)
    logger.info("Saved untrained hidden states to %s", untrained_path)

    logger.info(
        "Trained shape: %s  |  Untrained shape: %s",
        trained_hidden.shape, untrained_hidden.shape,
    )


# ---------------------------------------------------------------------------
# Extract all models
# ---------------------------------------------------------------------------

def extract_all(
    data_dir: Optional[str] = None,
    surrogate_dir: Optional[str] = None,
) -> None:
    """Extract hidden states for all LSTM sizes and the TCN.

    For each LSTM hidden size in ``HIDDEN_SIZES`` = [64, 128, 256]:
        - Trained:   ``{surrogate_dir}/hidden/lstm_{h}_trained.npz``
        - Untrained: ``{surrogate_dir}/hidden/lstm_{h}_untrained.npz``

    For the TCN:
        - Trained:   ``{surrogate_dir}/hidden/tcn_trained.npz``
        - Untrained: ``{surrogate_dir}/hidden/tcn_untrained.npz``

    Parameters
    ----------
    data_dir : str or Path, optional
        Trial data directory.  Defaults to ``config.BAHL_TRIAL_DIR``.
    surrogate_dir : str or Path, optional
        Base directory for model checkpoints and hidden state outputs.
        Defaults to ``config.SURROGATE_DIR``.
    """
    if data_dir is None:
        data_dir = str(BAHL_TRIAL_DIR)
    if surrogate_dir is None:
        surrogate_dir = str(SURROGATE_DIR)

    surrogate_dir = Path(surrogate_dir)
    hidden_dir = surrogate_dir / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)

    # --- LSTMs ---
    for h in HIDDEN_SIZES:
        # Try both naming conventions from training
        model_path = surrogate_dir / f"lstm_h{h}_best.pt"
        if not model_path.exists():
            model_path = surrogate_dir / f"lstm_{h}.pt"
        if not model_path.exists():
            logger.warning(
                "Checkpoint not found for LSTM h=%d -- skipping", h
            )
            continue

        logger.info("=== LSTM hidden_size=%d ===", h)
        extract_and_save(
            model_class=L5PC_LSTM,
            model_path=str(model_path),
            test_data_dir=data_dir,
            trained_path=str(hidden_dir / f"lstm_{h}_trained.npz"),
            untrained_path=str(hidden_dir / f"lstm_{h}_untrained.npz"),
            input_dim=TOTAL_SYN,
            hidden_size=h,
            n_layers=N_LSTM_LAYERS,
        )

    # --- TCN ---
    tcn_path = surrogate_dir / "tcn.pt"
    if tcn_path.exists():
        logger.info("=== TCN ===")
        extract_and_save(
            model_class=L5PC_TCN,
            model_path=str(tcn_path),
            test_data_dir=data_dir,
            trained_path=str(hidden_dir / "tcn_trained.npz"),
            untrained_path=str(hidden_dir / "tcn_untrained.npz"),
            input_dim=TOTAL_SYN,
            n_layers=TCN_N_LAYERS,
            n_features=TCN_N_FEATURES,
        )
    else:
        logger.warning("Checkpoint not found: %s -- skipping TCN", tcn_path)

    logger.info("=== All extractions complete ===")
