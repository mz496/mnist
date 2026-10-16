import dataclasses
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mnist.loss_landscape.animate.animate import Scene, build_scene, check_states, frame_counts, frame_label, load_sweep, read_frame_state, render_video, run_preview, set_frame
from mnist.loss_landscape.animate.snapshot_metrics_file_manager import SnapshotMetrics
from mnist.loss_landscape.animate.snapshot_surfaces_file_manager import SnapshotSurfaces
from mnist.loss_landscape.plot.grid_sweep_file_manager import GRID_SWEEP_SCHEMA_VERSION
from mnist.loss_landscape.plot.plot import sample_grid
from mnist.train import SNAPSHOT_SPACING


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
    assert "in every frame the contour at the latest path point reads the network's measured loss" in failed


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
