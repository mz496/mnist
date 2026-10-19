from pathlib import Path

import numpy as np
import pytest
import torch

from mnist.evaluate import evaluate
from mnist.loss_landscape.animate import snapshot_surfaces_file_manager as snapshot_surfaces
from mnist.loss_landscape.animate.snapshot_surfaces_file_manager import create_or_load_snapshot_surfaces
from mnist.loss_landscape.lib import get_pca_directions_for_snapshots, plane_base, project_snapshots_onto_pca_directions
from mnist.model import LeNet5


class FakeEvalData:
    """Snapshots, their PCA path, a small grid, and a tiny evaluation set, with the evaluation set and device stood in."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.chdir(tmp_path)
        parameter_count = sum(p.numel() for p in LeNet5().parameters())
        self.snapshots = np.random.default_rng(3).normal(0.0, 0.05, size=(6, parameter_count)).astype(np.float32)
        self.directions, _ = get_pca_directions_for_snapshots(self.snapshots)
        self.trajectory = project_snapshots_onto_pca_directions(self.snapshots, self.snapshots[-1], self.directions)
        self.alphas = np.array([-1.0, 0.0, 1.0])
        self.betas = np.array([-1.0, 0.0, 1.0])
        self.images = torch.rand(10, 1, 28, 28)
        self.labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.loads = 0

        def fake_snapshots(run_id: str) -> np.ndarray:
            return self.snapshots

        def fake_eval_tensors(split: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
            self.loads += 1
            return self.images, self.labels

        monkeypatch.setattr(snapshot_surfaces, "load_training_snapshots", fake_snapshots)
        monkeypatch.setattr(snapshot_surfaces, "load_eval_tensors", fake_eval_tensors)
        monkeypatch.setattr(snapshot_surfaces, "get_device", lambda: "cpu")


def test_surfaces_match_evaluating_the_shifted_weights_with_evaluate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = FakeEvalData(monkeypatch, tmp_path)
    frames = [2, 4, 6]
    surfaces = create_or_load_snapshot_surfaces("run", "train", data.trajectory, data.alphas, data.betas, frames)
    assert surfaces.loss.shape == (3, 3, 3) and surfaces.acc.shape == (3, 3, 3)

    # The independent route: build the weights for a grid point of a frame, run them through evaluate() over a loader
    model = LeNet5()
    loader = [(data.images, data.labels)]
    for f, n in enumerate(frames):
        base = plane_base(data.snapshots[n - 1], data.trajectory[n - 1], data.directions)
        for j, beta in enumerate(data.betas):
            for i, alpha in enumerate(data.alphas):
                weights = (base + alpha * data.directions[0] + beta * data.directions[1]).astype(np.float32)
                torch.nn.utils.vector_to_parameters(torch.from_numpy(weights), model.parameters())  # pyright: ignore[reportUnknownMemberType] Bad stubs from torch.
                loss, acc = evaluate(model, loader, "cpu")  # pyright: ignore[reportArgumentType] A list of batches stands in for a DataLoader.
                assert surfaces.loss[f, j, i] == pytest.approx(loss, abs=1e-4)
                assert surfaces.acc[f, j, i] == pytest.approx(acc, abs=1e-6)


def test_a_frames_network_sits_on_its_own_surface_at_its_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = FakeEvalData(monkeypatch, tmp_path)
    # Grid axes that contain each frame's own projected point, so the network itself is a node of its frame's grid
    frames = [3]
    point = data.trajectory[2]
    alphas = np.array([point[0] - 1.0, point[0], point[0] + 1.0])
    betas = np.array([point[1] - 1.0, point[1], point[1] + 1.0])
    surfaces = create_or_load_snapshot_surfaces("run", "train", data.trajectory, alphas, betas, frames)
    model = LeNet5()
    torch.nn.utils.vector_to_parameters(torch.from_numpy(data.snapshots[2].copy()), model.parameters())  # pyright: ignore[reportUnknownMemberType] Bad stubs from torch.
    measured, _ = evaluate(model, [(data.images, data.labels)], "cpu")  # pyright: ignore[reportArgumentType] A list of batches stands in for a DataLoader.
    assert surfaces.loss[0, 1, 1] == pytest.approx(measured, abs=1e-4)


def test_the_final_frame_is_the_plane_through_the_final_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = FakeEvalData(monkeypatch, tmp_path)
    surfaces = create_or_load_snapshot_surfaces("run", "train", data.trajectory, data.alphas, data.betas, [6])
    model = LeNet5()
    # The final snapshot is the center, at (0, 0), the middle node of this grid
    torch.nn.utils.vector_to_parameters(torch.from_numpy(data.snapshots[-1].copy()), model.parameters())  # pyright: ignore[reportUnknownMemberType] Bad stubs from torch.
    measured, _ = evaluate(model, [(data.images, data.labels)], "cpu")  # pyright: ignore[reportArgumentType] A list of batches stands in for a DataLoader.
    assert surfaces.loss[0, 1, 1] == pytest.approx(measured, abs=1e-4)


def test_saved_surfaces_are_reused_until_the_frames_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = FakeEvalData(monkeypatch, tmp_path)
    first = create_or_load_snapshot_surfaces("run", "train", data.trajectory, data.alphas, data.betas, [2, 4, 6])
    assert data.loads == 1
    assert snapshot_surfaces.surfaces_path("run") == Path("checkpoints") / "run" / "training_snapshot_surfaces.npz"
    again = create_or_load_snapshot_surfaces("run", "train", data.trajectory, data.alphas, data.betas, [2, 4, 6])
    assert data.loads == 1  # answered from the saved file
    assert np.array_equal(again.loss, first.loss) and np.array_equal(again.directions, first.directions)
    # Different frames, or different axes, are not what was saved
    create_or_load_snapshot_surfaces("run", "train", data.trajectory, data.alphas, data.betas, [3, 6])
    assert data.loads == 2
    create_or_load_snapshot_surfaces("run", "train", data.trajectory, data.alphas * 2.0, data.betas, [3, 6])
    assert data.loads == 3


def test_a_sweep_that_does_not_belong_to_the_snapshots_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = FakeEvalData(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="not the snapshots"):
        create_or_load_snapshot_surfaces("run", "train", data.trajectory + 1.0, data.alphas, data.betas, [2, 6])
    with pytest.raises(ValueError, match="frames must be between"):
        create_or_load_snapshot_surfaces("run", "train", data.trajectory, data.alphas, data.betas, [2, 7])
