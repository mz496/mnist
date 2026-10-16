from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mnist.config import load_default_config
from mnist.data import get_dataloaders
from mnist.evaluate import evaluate
from mnist.loss_landscape.file_io_utils import read_arrays
from mnist.model import LeNet5
from mnist.train import load_training_snapshots
from mnist.utils import get_device


@dataclass
class SnapshotMetrics:
    """The loss and accuracy of the network at each training snapshot, measured by running it, not read off a grid."""
    loss: np.ndarray  # [n]
    acc: np.ndarray  # [n]


def metrics_path(run_id: str) -> Path:
    return Path("checkpoints") / run_id / "training_snapshot_metrics.npz"


def load_snapshot_metrics(saved: dict[str, Any], split: str, n: int) -> SnapshotMetrics | None:
    """The saved loss and accuracy of the first n snapshots on a split, or None if the saved values don't cover n snapshots."""
    if f"{split}_loss" in saved and f"{split}_acc" in saved and len(saved[f"{split}_loss"]) >= n:
        return SnapshotMetrics(loss=saved[f"{split}_loss"][:n], acc=saved[f"{split}_acc"][:n])
    return None


def create_snapshot_metrics(run_id: str, split: str, n: int) -> SnapshotMetrics:
    """Run the network at each of the first n training snapshots of a run over the data of a split. Takes about half a second per snapshot."""
    snapshots = load_training_snapshots(run_id)
    if not 1 <= n <= len(snapshots):
        raise ValueError(f"asked for {n} snapshots, but run {run_id} has {len(snapshots)}")
    loaders = get_dataloaders(load_default_config().data)
    if split == "train":
        loader = loaders.train_subset
    elif split == "validation":
        loader = loaders.validation
    else:
        raise ValueError(f"can't evaluate on split {split!r}, expected 'train' or 'validation'")

    device = get_device()
    model = LeNet5().to(device)
    loss = np.zeros(n)
    acc = np.zeros(n)
    for i in range(n):
        # Put snapshot i's flat weight vector back into the model, the inverse of how the snapshots were flattened
        flat = torch.from_numpy(snapshots[i].copy()).to(device)  # pyright: ignore[reportUnknownMemberType] Bad stubs from torch.
        torch.nn.utils.vector_to_parameters(flat, model.parameters())
        loss[i], acc[i] = evaluate(model, loader, device)
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"Evaluated the network at snapshot {i + 1}/{n}")
    return SnapshotMetrics(loss=loss, acc=acc)


def create_or_load_snapshot_metrics(run_id: str, split: str, n: int) -> SnapshotMetrics:
    """
    The network's loss and accuracy at each of the first n training snapshots of a run, over the data a sweep was measured on.
    They are saved in training_snapshot_metrics.npz under the keys <split>_loss and <split>_acc, and reused as long as they
    cover at least n snapshots. Each split keeps its own keys, so measuring one never discards another.
    """
    cache = metrics_path(run_id)
    saved = read_arrays(cache)
    loaded = load_snapshot_metrics(saved, split, n)
    if loaded is not None:
        return loaded

    metrics = create_snapshot_metrics(run_id, split, n)
    saved[f"{split}_loss"] = metrics.loss
    saved[f"{split}_acc"] = metrics.acc
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **saved)
    return metrics
