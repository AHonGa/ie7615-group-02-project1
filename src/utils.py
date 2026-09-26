"""Shared training/eval/plotting helpers used by both training notebooks, so the
custom CNN and the ResNet baseline are trained and measured identically (fair
comparison for the Milestone 1 rubric)."""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import confusion_matrix, classification_report
except ModuleNotFoundError:  # pragma: no cover - user may not have the optional eval dependency installed
    confusion_matrix = None
    classification_report = None

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    run_name: str,
    weight_decay: float = 1e-4,
    patience: int = 5,
):
    """Standard train/val loop with early stopping on val accuracy. Returns history
    dict and total wall-clock training time in seconds (for the comparison table)."""
    model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_state, epochs_no_improve = 0.0, None, 0

    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        train_loss, train_acc = running_loss / total, correct / total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                v_correct += (out.argmax(1) == y).sum().item()
                v_total += x.size(0)
        val_loss, val_acc = v_loss / v_total, v_correct / v_total
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(
            f"[{run_name}] epoch {epoch:>2}/{epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc, best_state, epochs_no_improve = val_acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{run_name}] early stopping at epoch {epoch} (best val_acc={best_val_acc:.3f})")
                break

    train_time_sec = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), LOGS_DIR / f"{run_name}_best.pt")
    with open(LOGS_DIR / f"{run_name}_history.json", "w") as f:
        json.dump({"history": history, "train_time_sec": train_time_sec, "best_val_acc": best_val_acc}, f, indent=2)

    plot_curves(history, run_name)
    return history, train_time_sec, best_val_acc


def plot_curves(history: dict, run_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{run_name} — loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{run_name} — accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(LOGS_DIR / f"{run_name}_curves.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, class_names=None):
    if confusion_matrix is None or classification_report is None:
        raise ModuleNotFoundError(
            "scikit-learn is required for evaluation metrics. Install it with `pip install scikit-learn`."
        )

    model.eval()
    model.to(device)
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.numpy().tolist())

    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    return {"accuracy": acc, "confusion_matrix": cm, "report": report, "preds": all_preds, "labels": all_labels}


def plot_confusion_matrix(cm, class_names, run_name: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{run_name} — confusion matrix (test)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(LOGS_DIR / f"{run_name}_confusion_matrix.png", dpi=150)
    plt.close(fig)


def model_size_mb(model) -> float:
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 4 / (1024 ** 2)  # float32 assumption
