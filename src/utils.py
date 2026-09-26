"""Utility helpers for training and evaluation."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"


def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device for training."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: torch.device,
    run_name: str,
    patience: int = 10,
) -> tuple[List[Dict[str, float]], float, float]:
    """Train a classification model and save best checkpoint/history.

    Returns a history list, training wall-clock time in seconds, and the best validation accuracy.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    history: List[Dict[str, float]] = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_seen = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            train_loss_total += loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            train_seen += batch_size

        train_loss = train_loss_total / max(train_seen, 1)
        train_acc = train_correct / max(train_seen, 1)

        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_seen = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)

                batch_size = inputs.size(0)
                val_loss_total += loss.item() * batch_size
                val_correct += (logits.argmax(dim=1) == targets).sum().item()
                val_seen += batch_size

        val_loss = val_loss_total / max(val_seen, 1)
        val_acc = val_correct / max(val_seen, 1)

        epoch_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(epoch_entry)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= patience:
            break

        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
        checkpoint = {"state_dict": best_state, "best_val_acc": best_val_acc, "history": history}
    else:
        checkpoint = {"state_dict": model.state_dict(), "best_val_acc": best_val_acc, "history": history}

    checkpoint_path = LOGS_DIR / f"{run_name}_best.pt"
    torch.save(checkpoint, checkpoint_path)

    history_path = LOGS_DIR / f"{run_name}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_plot = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_acc = [item["train_acc"] for item in history]
    val_acc = [item["val_acc"] for item in history]

    axes[0].plot(epochs_plot, train_loss, label="train")
    axes[0].plot(epochs_plot, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs_plot, train_acc, label="train")
    axes[1].plot(epochs_plot, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(LOGS_DIR / f"{run_name}_training_curves.png", dpi=200)
    plt.close(fig)

    return history, time.time() - start_time, best_val_acc
