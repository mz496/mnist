from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mnist.config import load_default_config
from mnist.data import get_dataloaders
from mnist.loss_landscape.file_io_utils import read_arrays
from mnist.loss_landscape.lib import get_pca_directions_for_snapshots, plane_base, project_snapshots_onto_pca_directions
from mnist.model import LeNet5
from mnist.train import load_training_snapshots
from mnist.utils import get_device


@dataclass
class SnapshotSurfaces:
    """
    For each frame of an animation, the network's loss and accuracy on the grid of (alpha, beta) points of the PCA plane,
    shifted to pass through the network at that frame's training step. The directions and the (alpha, beta) axes are the
    same in every frame. Only the plane's offset changes.
    """
    loss: np.ndarray  # [frames, len(betas), len(alphas)]
    acc: np.ndarray  # [frames, len(betas), len(alphas)]
    frames: np.ndarray  # [frames], how many training snapshots each frame shows
    directions: np.ndarray  # [2, D], the PCA directions the grids were laid out along
    direction_error: float  # largest difference between projecting the snapshots on `directions` and the sweep's saved path


def surfaces_path(run_id: str) -> Path:
    return Path("checkpoints") / run_id / "training_snapshot_surfaces.npz"


# How far the saved path may differ from the snapshots projected on the directions before the two are taken to disagree.
# The path is on the order of ten units across.
DIRECTION_TOLERANCE = 1e-3


def direction_error(snapshots: np.ndarray, directions: np.ndarray, trajectory: np.ndarray) -> float:
    """How far projecting the snapshots on the directions, centered on the last one, is from a sweep's saved path."""
    projected = project_snapshots_onto_pca_directions(snapshots, snapshots[-1], directions)
    return float(np.abs(projected - trajectory).max())


def load_eval_tensors(split: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The images and labels of the data a sweep was measured on, as two tensors on the device, preprocessed like evaluate sees them."""
    loaders = get_dataloaders(load_default_config().data)
    if split == "train":
        loader = loaders.train_subset
    elif split == "validation":
        loader = loaders.validation
    else:
        raise ValueError(f"can't evaluate on split {split!r}, expected 'train' or 'validation'")
    batches = list(loader)
    return torch.cat([images for images, _ in batches]).to(device), torch.cat([labels for _, labels in batches]).to(device)


def load_snapshot_surfaces(saved: dict[str, Any], split: str, snapshots: np.ndarray, trajectory: np.ndarray, alphas: np.ndarray, betas: np.ndarray, frames: list[int]) -> SnapshotSurfaces | None:
    """The saved surfaces of a split, or None if there are none, or they were made for other frames, axes or directions."""
    keys = [f"{split}_{name}" for name in ("loss", "acc", "frames", "alphas", "betas", "directions")]
    if not all(key in saved for key in keys):
        return None
    directions = saved[f"{split}_directions"]
    error = direction_error(snapshots, directions, trajectory)
    if (
        np.array_equal(saved[f"{split}_frames"], np.array(frames))
        and np.array_equal(saved[f"{split}_alphas"], alphas)
        and np.array_equal(saved[f"{split}_betas"], betas)
        and error <= DIRECTION_TOLERANCE
    ):
        return SnapshotSurfaces(loss=saved[f"{split}_loss"], acc=saved[f"{split}_acc"], frames=saved[f"{split}_frames"], directions=directions, direction_error=error)
    return None


def create_snapshot_surfaces(run_id: str, split: str, snapshots: np.ndarray, trajectory: np.ndarray, alphas: np.ndarray, betas: np.ndarray, frames: list[int]) -> SnapshotSurfaces:
    """
    Run the network at every point of the (alpha, beta) grid of the plane through snapshot n, parallel to the PCA plane, for
    each frame (a count n of training snapshots). The whole evaluation set goes through the network as one batch for each grid
    point, which is about 40 times faster than looping over a data loader and gives the same numbers.

    The directions are worked out again from the snapshots, since the sweep file doesn't keep them. They are checked
    against the sweep's saved path, so a sweep that belongs to different snapshots is refused.
    """
    directions, _ = get_pca_directions_for_snapshots(snapshots)
    error = direction_error(snapshots, directions, trajectory)
    if error > DIRECTION_TOLERANCE:
        raise ValueError(f"the sweep's path is not the snapshots of run {run_id} projected on their PCA directions (largest difference {error:.3g}). It belongs to a different run or snapshots.")

    device = get_device()
    images, labels = load_eval_tensors(split, device)
    model = LeNet5().to(device)
    model.eval()
    d = torch.from_numpy(directions.astype(np.float32)).to(device)  # pyright: ignore[reportUnknownMemberType] Bad stubs from torch.
    loss = np.zeros((len(frames), len(betas), len(alphas)))
    acc = np.zeros((len(frames), len(betas), len(alphas)))
    with torch.no_grad():
        for f, n in enumerate(frames):
            # The plane through snapshot n, parallel to the PCA plane: snapshot n itself is at its own projected (alpha, beta)
            base = torch.from_numpy(plane_base(snapshots[n - 1], trajectory[n - 1], directions).astype(np.float32)).to(device)  # pyright: ignore[reportUnknownMemberType] Bad stubs from torch.
            # Collect on the device and read back once per frame: a read back for every grid point would stall the device
            losses: list[torch.Tensor] = []
            accuracies: list[torch.Tensor] = []
            for beta in betas:
                for alpha in alphas:
                    torch.nn.utils.vector_to_parameters(base + float(alpha) * d[0] + float(beta) * d[1], model.parameters())
                    logits = model(images)
                    losses.append(F.cross_entropy(logits, labels))
                    accuracies.append((logits.argmax(dim=1) == labels).float().mean())
            loss[f] = torch.stack(losses).cpu().numpy().reshape(len(betas), len(alphas))
            acc[f] = torch.stack(accuracies).cpu().numpy().reshape(len(betas), len(alphas))
            if (f + 1) % 10 == 0 or f + 1 == len(frames):
                print(f"Evaluated the surface around the network at frame {f + 1}/{len(frames)} (snapshot {n})")
    return SnapshotSurfaces(loss=loss, acc=acc, frames=np.array(frames), directions=directions, direction_error=error)


def create_or_load_snapshot_surfaces(run_id: str, split: str, trajectory: np.ndarray, alphas: np.ndarray, betas: np.ndarray, frames: list[int]) -> SnapshotSurfaces:
    """
    The loss and accuracy of the network on each frame's grid. A frame is a few seconds of evaluation, so the result is saved in
    training_snapshot_surfaces.npz and reused when the frames, the axes and the directions all still match.
    """
    snapshots = load_training_snapshots(run_id)
    n_points = len(trajectory)
    if n_points > len(snapshots):
        raise ValueError(f"the sweep's path has {n_points} points but run {run_id} has only {len(snapshots)} snapshots")
    snapshots = snapshots[:n_points]
    if not frames or min(frames) < 1 or max(frames) > n_points:
        raise ValueError(f"frames must be between 1 and {n_points}")

    cache = surfaces_path(run_id)
    saved = read_arrays(cache)
    loaded = load_snapshot_surfaces(saved, split, snapshots, trajectory, alphas, betas, frames)
    if loaded is not None:
        return loaded

    surfaces = create_snapshot_surfaces(run_id, split, snapshots, trajectory, alphas, betas, frames)
    saved[f"{split}_loss"] = surfaces.loss
    saved[f"{split}_acc"] = surfaces.acc
    saved[f"{split}_frames"] = surfaces.frames
    saved[f"{split}_alphas"] = alphas
    saved[f"{split}_betas"] = betas
    saved[f"{split}_directions"] = surfaces.directions
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **saved)
    return surfaces
