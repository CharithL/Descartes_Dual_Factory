"""
L5PC DESCARTES -- Surrogate Training Loop

AdamW optimiser with ReduceLROnPlateau scheduler and early stopping.
MSE loss on somatic voltage targets (normalised to zero mean / unit std
before computing loss, so gradient magnitudes are independent of the
target's physical scale ~[-80, +40] mV).

All hyperparameters imported from ``l5pc.config``.

Usage
-----
    from l5pc.surrogates.train import train_lstm, train_tcn

    # Train an LSTM with hidden_size=128
    model = train_lstm(hidden_size=128, data_dir='data/bahl_trials',
                       save_path='data/surrogates/lstm_128.pt')

    # Train a TCN from scratch
    model = train_tcn(data_dir='data/bahl_trials',
                      save_path='data/surrogates/tcn.pt')
"""

import copy
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from l5pc.config import (
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    HIDDEN_SIZES,
    LEARNING_RATE,
    LR_FACTOR,
    LR_PATIENCE,
    MAX_EPOCHS,
    N_LSTM_LAYERS,
    SURROGATE_DIR,
    TCN_N_FEATURES,
    TCN_N_LAYERS,
    TOTAL_SYN,
    TRAIN_SPLIT,
    VAL_SPLIT,
    WEIGHT_DECAY,
)
from l5pc.surrogates.lstm import L5PC_LSTM
from l5pc.surrogates.tcn import L5PC_TCN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def create_dataset(
    trial_dir: str,
    split: str = "train",
) -> TensorDataset:
    """Load pre-saved trial data and return a ``TensorDataset``.

    Expected directory layout::

        trial_dir/
            inputs.npy    -- shape (N_TRIALS, T, input_dim)
            outputs.npy   -- shape (N_TRIALS, T)

    The first ``TRAIN_SPLIT`` trials are training, the next ``VAL_SPLIT``
    are validation, and the remaining are test.

    Parameters
    ----------
    trial_dir : str or Path
        Directory containing ``inputs.npy`` and ``outputs.npy``.
    split : str
        One of ``'train'``, ``'val'``, ``'test'``.

    Returns
    -------
    dataset : TensorDataset
        Contains (inputs, outputs) tensors as float32.
    """
    trial_dir = Path(trial_dir)

    # Try aggregated .npy files first, fall back to per-trial .npz
    if (trial_dir / "inputs.npy").exists():
        inputs = np.load(trial_dir / "inputs.npy")    # (N, T, C)
        outputs = np.load(trial_dir / "outputs.npy")   # (N, T)
    else:
        # Load individual trial_XXX.npz files (from simulation output)
        trial_files = sorted(trial_dir.glob("trial_*.npz"))
        if not trial_files:
            raise FileNotFoundError(
                f"No trial data in {trial_dir}. "
                "Expected inputs.npy/outputs.npy or trial_XXX.npz files."
            )
        input_list, output_list = [], []
        for fpath in trial_files:
            data = np.load(fpath)
            input_list.append(data['inputs'])    # (T, C)
            output_list.append(data['output'])   # (T,)
        inputs = np.stack(input_list, axis=0)    # (N, T, C)
        outputs = np.stack(output_list, axis=0)  # (N, T)
        logger.info("Loaded %d trials from individual .npz files", len(inputs))

    if split == "train":
        start, end = 0, TRAIN_SPLIT
    elif split == "val":
        start, end = TRAIN_SPLIT, TRAIN_SPLIT + VAL_SPLIT
    elif split == "test":
        start, end = TRAIN_SPLIT + VAL_SPLIT, len(inputs)
    else:
        raise ValueError(f"Unknown split: {split!r}. Use 'train', 'val', or 'test'.")

    x = torch.tensor(inputs[start:end], dtype=torch.float32)
    y = torch.tensor(outputs[start:end], dtype=torch.float32)

    logger.info(
        "Loaded %s split: %d trials, input shape %s, output shape %s",
        split, len(x), tuple(x.shape), tuple(y.shape),
    )
    return TensorDataset(x, y)


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_path: str,
    config: Optional[dict] = None,
) -> nn.Module:
    """Train a surrogate model with AdamW, LR scheduling, and early stopping.

    Parameters
    ----------
    model : nn.Module
        Model with ``forward(x)`` returning ``(output,)`` where output
        has shape ``(batch, T)``.
    train_loader : DataLoader
        Training data loader yielding ``(inputs, targets)`` batches.
    val_loader : DataLoader
        Validation data loader.
    save_path : str or Path
        Where to save the best model checkpoint.
    config : dict, optional
        Override default hyperparameters.  Recognised keys:
        ``lr``, ``weight_decay``, ``lr_patience``, ``lr_factor``,
        ``early_stop_patience``, ``max_epochs``.

    Returns
    -------
    model : nn.Module
        Model loaded with best-validation-loss weights.
    """
    cfg = {
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "lr_patience": LR_PATIENCE,
        "lr_factor": LR_FACTOR,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "max_epochs": MAX_EPOCHS,
    }
    if config:
        cfg.update(config)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimiser = AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimiser,
        mode="min",
        patience=cfg["lr_patience"],
        factor=cfg["lr_factor"],
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    # ---- Target normalisation ----
    # Raw voltage targets span ~[-80, +40] mV, giving initial MSE ~ 4000.
    # Normalising to zero-mean / unit-std makes the loss O(1) and
    # stabilises gradients for all model sizes (especially h=64/256).
    # Compute statistics from the *training* loader targets only.
    _all_y = []
    for _, yb in train_loader:
        _all_y.append(yb)
    _all_y = torch.cat(_all_y, dim=0)
    y_mean = _all_y.mean().item()
    y_std = _all_y.std().item()
    if y_std < 1e-8:
        y_std = 1.0
    del _all_y
    logger.info("Target normalisation: mean=%.3f  std=%.3f", y_mean, y_std)

    logger.info(
        "Training on %s | %d train batches, %d val batches | max %d epochs",
        device, len(train_loader), len(val_loader), cfg["max_epochs"],
    )

    for epoch in range(1, cfg["max_epochs"] + 1):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Normalise target to zero-mean / unit-std
            y_norm = (y_batch - y_mean) / y_std

            result = model(x_batch)
            # forward returns (output,) or (output, hidden)
            pred = result[0]

            loss = criterion(pred, y_norm)
            optimiser.zero_grad()
            loss.backward()
            # Gradient clipping: prevents exploding gradients in LSTM
            # recurrent path, especially important for small models (h=64).
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            train_loss_sum += loss.item() * x_batch.size(0)
            train_n += x_batch.size(0)

        train_loss = train_loss_sum / train_n

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                y_norm = (y_batch - y_mean) / y_std

                result = model(x_batch)
                pred = result[0]

                loss = criterion(pred, y_norm)
                val_loss_sum += loss.item() * x_batch.size(0)
                val_n += x_batch.size(0)

        val_loss = val_loss_sum / val_n
        scheduler.step(val_loss)

        current_lr = optimiser.param_groups[0]["lr"]
        logger.info(
            "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f  lr=%.2e",
            epoch, cfg["max_epochs"], train_loss, val_loss, current_lr,
        )

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_state, save_path)
            logger.info("  -> New best val_loss=%.6f, saved to %s", val_loss, save_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stop_patience"]:
                logger.info(
                    "Early stopping at epoch %d (patience=%d).",
                    epoch, cfg["early_stop_patience"],
                )
                break

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("Training complete. Best val_loss=%.6f", best_val_loss)
    return model


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def train_lstm(
    hidden_size: int = 128,
    data_dir: str = None,
    save_path: str = None,
    input_dim: int = TOTAL_SYN,
    config: Optional[dict] = None,
) -> L5PC_LSTM:
    """Train an LSTM surrogate end-to-end.

    Parameters
    ----------
    hidden_size : int
        LSTM hidden dimension (one of HIDDEN_SIZES = [64, 128, 256]).
    data_dir : str or Path
        Trial data directory.  Defaults to ``config.BAHL_TRIAL_DIR``.
    save_path : str or Path
        Model checkpoint path.  Defaults to
        ``SURROGATE_DIR / f'lstm_{hidden_size}.pt'``.
    input_dim : int
        Input channels per timestep.
    config : dict, optional
        Training hyperparameter overrides.

    Returns
    -------
    model : L5PC_LSTM
        Trained model with best-validation-loss weights.
    """
    from l5pc.config import BAHL_TRIAL_DIR

    if data_dir is None:
        data_dir = str(BAHL_TRIAL_DIR)
    if save_path is None:
        save_path = str(SURROGATE_DIR / f"lstm_{hidden_size}.pt")

    logger.info("=== Training LSTM (hidden_size=%d) ===", hidden_size)

    model = L5PC_LSTM(
        input_dim=input_dim,
        hidden_size=hidden_size,
        n_layers=N_LSTM_LAYERS,
    )
    logger.info("Model: %s", model)

    train_ds = create_dataset(data_dir, "train")
    val_ds = create_dataset(data_dir, "val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_model(model, train_loader, val_loader, save_path, config)


def train_tcn(
    data_dir: str = None,
    save_path: str = None,
    input_dim: int = TOTAL_SYN,
    config: Optional[dict] = None,
) -> L5PC_TCN:
    """Train a TCN surrogate from scratch.

    Parameters
    ----------
    data_dir : str or Path
        Trial data directory.  Defaults to ``config.BAHL_TRIAL_DIR``.
    save_path : str or Path
        Model checkpoint path.  Defaults to
        ``SURROGATE_DIR / 'tcn.pt'``.
    input_dim : int
        Input channels per timestep.
    config : dict, optional
        Training hyperparameter overrides.

    Returns
    -------
    model : L5PC_TCN
        Trained model with best-validation-loss weights.
    """
    from l5pc.config import BAHL_TRIAL_DIR

    if data_dir is None:
        data_dir = str(BAHL_TRIAL_DIR)
    if save_path is None:
        save_path = str(SURROGATE_DIR / "tcn.pt")

    logger.info("=== Training TCN ===")

    model = L5PC_TCN(
        input_dim=input_dim,
        n_layers=TCN_N_LAYERS,
        n_features=TCN_N_FEATURES,
    )
    logger.info("Model: %s", model)

    train_ds = create_dataset(data_dir, "train")
    val_ds = create_dataset(data_dir, "val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_model(model, train_loader, val_loader, save_path, config)
