from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import TRAIT_NAMES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    criterion = nn.MSELoss()
    model.eval()

    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        target = batch["target"]
        pred = model(batch)

        loss = criterion(pred, target)
        total_loss += float(loss.item()) * target.size(0)
        total_count += int(target.size(0))

        y_true.append(target.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())

    y_true_np = np.concatenate(y_true, axis=0)
    y_pred_np = np.concatenate(y_pred, axis=0)

    avg_loss = total_loss / max(total_count, 1)
    return y_true_np, y_pred_np, avg_loss


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(y_true - y_pred)
    mae_per_trait = abs_err.mean(axis=0)
    overall_mae = float(mae_per_trait.mean())

    metrics: Dict[str, float] = {
        "mae": overall_mae,
        "accuracy": float((1.0 - overall_mae) * 100.0),
    }

    for i, trait_name in enumerate(TRAIT_NAMES):
        trait_mae = float(mae_per_trait[i])
        metrics[f"mae_{trait_name}"] = trait_mae
        metrics[f"acc_{trait_name}"] = float((1.0 - trait_mae) * 100.0)

    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    criterion = nn.MSELoss()
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        target = batch["target"]

        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * target.size(0)
        total_count += int(target.size(0))

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    y_true, y_pred, avg_loss = _collect_predictions(model=model, loader=loader, device=device)
    metrics = compute_mae(y_true=y_true, y_pred=y_pred)
    metrics["loss"] = avg_loss
    return metrics


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, List[float]]:
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "valid_loss": [],
        "valid_mae": [],
        "valid_accuracy": [],
    }

    best_mae = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        valid_metrics = evaluate(model=model, loader=valid_loader, device=device)

        history["train_loss"].append(float(train_loss))
        history["valid_loss"].append(float(valid_metrics["loss"]))
        history["valid_mae"].append(float(valid_metrics["mae"]))
        history["valid_accuracy"].append(float(valid_metrics["accuracy"]))

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} | "
            f"valid_mae={valid_metrics['mae']:.4f} | "
            f"valid_acc={(1.0 - valid_metrics['mae']) * 100.0:.2f}"
        )

        if checkpoint_path is not None and valid_metrics["mae"] < best_mae:
            best_mae = float(valid_metrics["mae"])
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_valid_mae": best_mae,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

    return history


def save_history(history: Dict[str, List[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
