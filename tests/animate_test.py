import dataclasses
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mnist.loss_landscape.animate import (
    Scene,
    build_scene,
    frame_label,
    check_states,
    first_float,
    frame_counts,
    load_sweep,
    read_frame_state,
    render_video,
    run_preview,
    set_frame,
)
from mnist.evaluate import evaluate
from mnist.loss_landscape import snapshot_metrics_file_manager as snapshot_metrics, snapshot_surfaces_file_manager as snapshot_surfaces
from mnist.loss_landscape.lib import get_pca_directions_for_snapshots, plane_base, project_snapshots_onto_pca_directions
from mnist.loss_landscape.plot import sample_grid
from mnist.loss_landscape.grid_sweep_file_manager import GRID_SWEEP_SCHEMA_VERSION
from mnist.loss_landscape.snapshot_metrics_file_manager import SnapshotMetrics, create_or_load_snapshot_metrics
from mnist.loss_landscape.snapshot_surfaces_file_manager import SnapshotSurfaces, create_or_load_snapshot_surfaces
from mnist.model import LeNet5
from mnist.train import SNAPSHOT_SPACING
import torch


def write_sweep(path: Path, mode: str = "2d", with_trajectory: bool = True, n_points: int = 40) -> Path:
    """A small 2d trajectory sweep file with the keys grid_sweep writes: a smooth bowl, and a path that ends at the origin."""
    alphas = np.linspace(-6.0, 1.0, 9)
    betas = np.linspace(-2.0, 3.0, 7)
    alpha_grid, beta_grid = np.meshgrid(alphas, betas)  # [len(betas), len(alphas)]
    # Typed as Any because numpy's savez stubs would otherwise read a **dict[str, ndarray] as possibly meant for its allow_pickle keyword
    arrays: dict[str, Any] = {
        "alphas": alphas,
        "betas": betas,
        "loss": 0.05 + 0.04 * (alpha_grid**2 + 2 * beta_grid**2),
        "acc": 1.0 - 0.3 * (1.0 - np.exp(-0.1 * (alpha_grid**2 + beta_grid**2))),
        "seed": np.array(0),
        "run_id": np.array("test-run"),
        "split": np.array("train"),
        "mode": np.array(mode),
        "schema_version": np.array(GRID_SWEEP_SCHEMA_VERSION),
    }
    if with_trajectory:
        t = np.linspace(0.0, 1.0, n_points)[:, None]
        arrays["trajectory"] = np.hstack([-5.0 * (1 - t), 2.0 * (1 - t) ** 2])  # starts at (-5, 2), ends at (0, 0)
        arrays["explained_variance"] = np.array([0.8, 0.1])
    np.savez(path, **arrays)
    return path


def fake_metrics(n: int) -> SnapshotMetrics:
    """Measured values that are clearly not what the contour grid would read: the loss falls from 2.3, the accuracy rises from 10%."""
    return SnapshotMetrics(loss=np.linspace(2.3, 0.02, n), acc=np.linspace(0.10, 0.99, n))


def surfaces_for(sweep_path: Path, counts: list[int]) -> tuple[SnapshotSurfaces, SnapshotMetrics]:
    """
    Surfaces that differ from frame to frame but end at the sweep's own grid, with measured losses that agree with what each
    surface reads at its frame's dot, as they do for a real run.
    """
    sweep = load_sweep(sweep_path)
    n_points = len(sweep.trajectory)
    frames = len(counts)
    loss = np.stack([sweep.loss * (1.0 + 0.4 * (frames - 1 - f) / frames) for f in range(frames)])
    acc = np.stack([sweep.acc * (1.0 - 0.3 * (frames - 1 - f) / frames) for f in range(frames)])
    measured_loss = np.full(n_points, 0.5)
    for f, n in enumerate(counts):
        measured_loss[n - 1] = sample_grid(sweep.alphas, sweep.betas, loss[f], sweep.trajectory[n - 1:n])[0]
    metrics = SnapshotMetrics(loss=measured_loss, acc=np.full(n_points, 0.9))
    surfaces = SnapshotSurfaces(loss=loss, acc=acc, frames=np.array(counts), directions=np.zeros((2, 3)), direction_error=0.0)
    return surfaces, metrics


@pytest.fixture
def scene(tmp_path: Path):
    sweep_path = write_sweep(tmp_path / "sweep.npz")
    sweep = load_sweep(sweep_path)
    counts = frame_counts(len(sweep.trajectory), 10)
    surfaces, metrics = surfaces_for(sweep_path, counts)
    scene = build_scene(sweep, "loss", metrics, surfaces, counts)
    yield scene
    plt.close(scene.fig)


def test_frame_counts_end_on_the_whole_run():
    counts = frame_counts(860, 10)
    assert len(counts) == 86
    assert counts[0] == 10 and counts[-1] == 860
    assert frame_counts(25, 10) == [10, 20, 25]
    assert frame_counts(5, 10) == [5]
    with pytest.raises(ValueError):
        frame_counts(100, 0)


def test_set_frame_shows_the_first_n_points(scene: Scene):
    set_frame(scene, 20)
    assert np.array_equal(np.array(scene.artists.path.get_offsets()), scene.trajectory[:20])
    assert (first_float(scene.highlight.get_xdata()), first_float(scene.highlight.get_ydata())) == (float(scene.trajectory[19, 0]), float(scene.trajectory[19, 1]))
    assert scene.label.get_text().splitlines()[0] == f"step {20 * SNAPSHOT_SPACING}"
    with pytest.raises(ValueError):
        set_frame(scene, 0)


def test_scales_do_not_change_between_frames(scene: Scene):
    states = [read_frame_state(scene, n) for n in (10, 20, 40)]
    failed = [c.name for c in check_states(states, scene) if not c.ok]
    assert failed == []


def test_preview_passes_and_writes_three_stills(scene: Scene, tmp_path: Path):
    counts = frame_counts(len(scene.trajectory), 10)
    assert run_preview(scene, counts, tmp_path / "animations", "stem", "stamp")
    assert sorted(p.name for p in (tmp_path / "animations").iterdir()) == [
        "stamp_stem_preview=true_frame=1.png",
        "stamp_stem_preview=true_frame=2.png",
        "stamp_stem_preview=true_frame=4.png",
    ]


def test_checks_catch_a_frame_whose_axis_limits_moved(scene: Scene):
    states = [read_frame_state(scene, n) for n in (10, 20, 40)]
    states[1] = dataclasses.replace(states[1], xlim=(states[1].xlim[0] + 1.0, states[1].xlim[1]))
    failed = [c.name for c in check_states(states, scene) if not c.ok]
    assert "axis limits identical across frames" in failed


def test_checks_catch_a_drifting_dot_color_range(scene: Scene):
    states = [read_frame_state(scene, n) for n in (10, 20, 40)]
    low, high = states[2].dots_clim
    states[2] = dataclasses.replace(states[2], dots_clim=(low, high * 2))
    failed = [c.name for c in check_states(states, scene) if not c.ok]
    assert "dot color range identical across frames" in failed


def test_checks_catch_the_wrong_number_of_points(scene: Scene):
    states = [read_frame_state(scene, n) for n in (10, 20, 40)]
    states[0] = dataclasses.replace(states[0], offsets=states[0].offsets[:0])
    failed = [c.name for c in check_states(states, scene) if not c.ok]
    assert "frame f draws exactly the first n path points" in failed


def test_a_sweep_without_a_trajectory_is_rejected(tmp_path: Path):
    path = write_sweep(tmp_path / "no_trajectory.npz", with_trajectory=False)
    with pytest.raises(ValueError, match="trajectory"):
        load_sweep(path)


def test_a_1d_sweep_is_rejected(tmp_path: Path):
    path = write_sweep(tmp_path / "wrong_mode.npz", mode="1d")
    with pytest.raises(ValueError, match="2d sweep"):
        load_sweep(path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_a_video_is_written(scene: Scene, tmp_path: Path):
    out_path = tmp_path / "out" / "anim.mp4"
    render_video(scene, frame_counts(len(scene.trajectory), 10), 5, out_path)
    assert out_path.is_file() and out_path.stat().st_size > 0


def test_dots_are_shaded_by_the_measured_loss_not_the_grid(scene: Scene):
    assert scene.metrics is not None
    assert np.array_equal(scene.trajectory_loss, scene.metrics.loss)
    set_frame(scene, 20)
    assert np.allclose(np.array(scene.artists.path.get_array()), scene.metrics.loss[:20])
    assert scene.artists.path.get_clim() == (float(scene.metrics.loss.min()), float(scene.metrics.loss.max()))


def test_each_frame_prints_the_measured_loss_and_accuracy_for_its_step(scene: Scene):
    assert scene.metrics is not None
    set_frame(scene, 10)
    lines = scene.label.get_text().splitlines()
    assert lines[0] == f"step {10 * SNAPSHOT_SPACING}"
    assert lines[1] == f"training loss {scene.metrics.loss[9]:.3f}"
    assert lines[2] == f"training accuracy {scene.metrics.acc[9]:.1%}"
    assert frame_label(scene, 40) != frame_label(scene, 10)


def test_checks_catch_a_printed_loss_that_is_not_the_measured_one(scene: Scene):
    states = [read_frame_state(scene, n) for n in (10, 20, 40)]
    states[1] = dataclasses.replace(states[1], label=states[1].label.replace("loss ", "loss 9"))
    failed = [c.name for c in check_states(states, scene) if not c.ok]
    assert "the loss and accuracy printed in each frame are the network's measured values at that step" in failed


def test_a_measurement_for_every_snapshot_is_required(tmp_path: Path):
    sweep_path = write_sweep(tmp_path / "sweep.npz")
    sweep = load_sweep(sweep_path)
    counts = frame_counts(len(sweep.trajectory), 10)
    with pytest.raises(ValueError, match="measured snapshots"):
        build_scene(sweep, "loss", fake_metrics(len(sweep.trajectory) - 1), *surfaces_for(sweep_path, counts)[:1], counts)


def test_evaluating_snapshots_runs_the_network_and_reuses_the_saved_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    parameter_count = sum(p.numel() for p in LeNet5().parameters())
    generator = np.random.default_rng(0)
    snapshots = generator.normal(0.0, 0.05, size=(4, parameter_count)).astype(np.float32)
    batch = (torch.rand(6, 1, 28, 28), torch.tensor([0, 1, 2, 3, 4, 5]))

    class FakeLoaders:
        train_subset = [batch]
        validation = [batch]

    def fake_snapshots(run_id: str) -> np.ndarray:
        return snapshots

    def fake_loaders(data_config: object) -> FakeLoaders:
        return FakeLoaders()

    # The working directory is the temporary one, where there is no configs/default.yaml to read
    monkeypatch.setattr(snapshot_metrics, "load_default_config", lambda: SimpleNamespace(data=None))
    monkeypatch.setattr(snapshot_metrics, "load_training_snapshots", fake_snapshots)
    monkeypatch.setattr(snapshot_metrics, "get_dataloaders", fake_loaders)
    monkeypatch.setattr(snapshot_metrics, "get_device", lambda: "cpu")

    first = create_or_load_snapshot_metrics("run", "train", 4)
    assert first.loss.shape == (4,) and first.acc.shape == (4,)
    assert np.all(np.isfinite(first.loss)) and np.all((first.acc >= 0) & (first.acc <= 1))
    # Different weights give different losses, which shows each snapshot is really loaded into the network
    assert len(set(np.round(first.loss, 6))) == 4
    # Saved next to the snapshots, in the file named like training_snapshot_vectors.npz
    assert snapshot_metrics.metrics_path("run") == Path("checkpoints") / "run" / "training_snapshot_metrics.npz"
    assert snapshot_metrics.metrics_path("run").is_file()

    # A second call is answered from the saved file: running the network again would raise
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the network was evaluated again")

    monkeypatch.setattr(snapshot_metrics, "evaluate", refuse)
    again = create_or_load_snapshot_metrics("run", "train", 4)
    assert np.array_equal(again.loss, first.loss)
    assert np.array_equal(create_or_load_snapshot_metrics("run", "train", 2).loss, first.loss[:2])


def test_each_split_keeps_its_own_results_in_the_one_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    parameter_count = sum(p.numel() for p in LeNet5().parameters())
    snapshots = np.random.default_rng(1).normal(0.0, 0.05, size=(3, parameter_count)).astype(np.float32)
    train_batch = (torch.rand(6, 1, 28, 28), torch.tensor([0, 1, 2, 3, 4, 5]))
    validation_batch = (torch.rand(6, 1, 28, 28), torch.tensor([5, 4, 3, 2, 1, 0]))

    class FakeLoaders:
        train_subset = [train_batch]
        validation = [validation_batch]

    def fake_snapshots(run_id: str) -> np.ndarray:
        return snapshots

    def fake_loaders(data_config: object) -> FakeLoaders:
        return FakeLoaders()

    monkeypatch.setattr(snapshot_metrics, "load_default_config", lambda: SimpleNamespace(data=None))
    monkeypatch.setattr(snapshot_metrics, "load_training_snapshots", fake_snapshots)
    monkeypatch.setattr(snapshot_metrics, "get_dataloaders", fake_loaders)
    monkeypatch.setattr(snapshot_metrics, "get_device", lambda: "cpu")

    train = create_or_load_snapshot_metrics("run", "train", 3)
    validation = create_or_load_snapshot_metrics("run", "validation", 3)
    with np.load(snapshot_metrics.metrics_path("run")) as saved:
        assert sorted(saved.files) == ["train_acc", "train_loss", "validation_acc", "validation_loss"]
        # Measuring validation did not discard the train results
        assert np.array_equal(saved["train_loss"], train.loss)
        assert np.array_equal(saved["validation_loss"], validation.loss)


# ---- The contour evaluated again for each frame ----


def test_each_frame_draws_its_own_surface(scene: Scene):
    assert scene.grids is not None
    for n in (10, 20, 40):
        set_frame(scene, n)
        assert np.array_equal(scene.drawn_values, scene.grids[scene.counts.index(n)])
    # The surfaces really differ from frame to frame, so a redraw is not a no-op
    assert not np.array_equal(scene.grids[0], scene.grids[-1])


def test_the_levels_cover_every_frame_and_do_not_change(scene: Scene):
    assert scene.grids is not None
    levels = np.array(scene.filled.levels)
    assert levels[0] <= scene.grids.min() and levels[-1] >= scene.grids.max()
    for n in (10, 40):
        set_frame(scene, n)
        assert np.array_equal(np.array(scene.filled.levels), levels)


def test_a_frame_without_an_evaluated_surface_is_refused(scene: Scene):
    with pytest.raises(ValueError, match="no surface was evaluated"):
        set_frame(scene, 7)


def test_surfaces_for_other_frames_are_refused(tmp_path: Path):
    sweep_path = write_sweep(tmp_path / "sweep.npz")
    sweep = load_sweep(sweep_path)
    counts = frame_counts(len(sweep.trajectory), 10)
    surfaces, metrics = surfaces_for(sweep_path, counts)
    with pytest.raises(ValueError, match="different frames"):
        build_scene(sweep, "loss", metrics, surfaces, frame_counts(len(sweep.trajectory), 20))


def test_preview_passes_with_redrawn_contours(scene: Scene, tmp_path: Path):
    assert run_preview(scene, scene.counts, tmp_path / "animations", "stem", "stamp")


def test_checks_catch_a_contour_that_does_not_read_the_measured_loss(scene: Scene):
    assert scene.metrics is not None
    wrong = dataclasses.replace(scene.metrics, loss=scene.metrics.loss + 1.0)
    broken = dataclasses.replace(scene, metrics=wrong)
    states = [read_frame_state(broken, n) for n in (10, 20, 40)]
    failed = [c.name for c in check_states(states, broken) if not c.ok]
    assert "in every frame the contour at the highlighted point reads the network's measured loss" in failed


def test_checks_catch_a_last_surface_that_is_not_the_sweeps_grid(scene: Scene):
    assert scene.surfaces is not None
    shifted = dataclasses.replace(scene.surfaces, loss=scene.surfaces.loss + 0.1)
    broken = dataclasses.replace(scene, surfaces=shifted)
    states = [read_frame_state(broken, n) for n in (10, 20, 40)]
    failed = [c.name for c in check_states(states, broken) if not c.ok]
    assert "the last frame's surface equals the sweep's saved grid, which is the plane through the final weights" in failed


def test_checks_catch_directions_that_do_not_match_the_saved_path(scene: Scene):
    assert scene.surfaces is not None
    off = dataclasses.replace(scene.surfaces, direction_error=0.5)
    broken = dataclasses.replace(scene, surfaces=off)
    states = [read_frame_state(broken, n) for n in (10, 20, 40)]
    failed = [c.name for c in check_states(states, broken) if not c.ok]
    assert "the directions the surfaces lie along reproduce the sweep's saved path" in failed


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
