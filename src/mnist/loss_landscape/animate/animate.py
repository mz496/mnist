# matplotlib ships partial type information, so strict mode flags its calls as partially unknown.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

"""
Animate a training run's path across the loss surface of a PCA plane of its training snapshots.

## The plane

- The two axes are the top two PCA directions as unit direction vectors d1, d2 in weight-space of all training snapshots in the video's window.
    - Note that this can be either the full run, or some prefix via --max-step.
- The origin (0, 0) is the final snapshot of the window, so the path always ends there.
- The range (min/max x/y) is fixed for every frame based on the origin + range of the points.

## The points

- These come from the grid_sweep npz file. Each point represents a training snapshot.
- A point's (u, v) coordinate is the snapshot minus the final snapshot, dotted with the two unit direction vectors, to plot it into the fixed plane.
  - ((u, v) are called alpha, beta in the plots)
- Loss values per-point are measured by running the snapshot's weights over the data, not read off the contour, as that is only correct for the current frame and point.
  - "Correct" is a bit lossy here since the grid resolution is low, but it would theoretically be correct.
  - Also note that we need to measure loss from the training examples, since the optimizer optimizes training loss.

## The contour

- In frame n, the grid point (u, v) contour value is the loss of the network with weights
base_n + u * d1 + v * d2, where base_n = snapshot_n - u_n * d1 - v_n * d2:
the plane parallel to the PCA plane that passes through snapshot n.
    
- This means that at frame n's own dot, the weights are snapshot n exactly, so the contour there reads the network's measured loss.
"""

import matplotlib

matplotlib.use("Agg")  # before pyplot is imported anywhere, so no window is ever opened

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import BoundaryNorm
from matplotlib.contour import QuadContourSet
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import offset_copy

from mnist.loss_landscape.file_io_utils import format_tags, viz_animations_dir, viz_filename
from mnist.loss_landscape.plot.grid_sweep import create_or_load_grid_sweep
from mnist.loss_landscape.plot.grid_sweep_file_manager import check_grid_sweep_schema_version, read_str, sweep_tags
from mnist.loss_landscape.animate.snapshot_metrics_file_manager import SnapshotMetrics, create_or_load_snapshot_metrics
from mnist.loss_landscape.animate.snapshot_surfaces_file_manager import DIRECTION_TOLERANCE, SnapshotSurfaces, create_or_load_snapshot_surfaces
from mnist.loss_landscape.plot.plot import (
    TrajectoryArtists,
    contour_levels,
    draw_contour_background,
    draw_trajectory_overlay,
    run_label,
    sample_grid,
    split_label,
)
from mnist.train import SNAPSHOT_SPACING

# The same resolution as the static plots. H.264 needs the pixel width and height to be even.
DPI = 150
FIGURE_WIDTH = 7.0

# The figure's height is worked out from the plot's shape, from these sizes in inches. Under the plot, from its bottom edge
# down: its x label, then the step readout and the legend, then the dots' colorbar with its tick labels and label.
TOP_MARGIN = 0.7
READOUT_OFFSET = 0.55
DOTS_COLORBAR_OFFSET = 1.45
DOTS_COLORBAR_HEIGHT = 0.2
BOTTOM_MARGIN = 0.4


def layout_figure(fig: Figure, ax: Axes, contour_colorbar: Colorbar, dots_colorbar: Colorbar) -> None:
    """
    Lay the figure out once, with the plot at the top, its contour colorbar as tall as it is, and the dots' colorbar a fixed
    distance under it, which leaves room between them for the readout and the legend. The height of the figure is chosen to fit.
    matplotlib's own layout can't do this, since it keeps a plot with a fixed aspect centered in the space it has.
    """
    fig.set_size_inches(FIGURE_WIDTH, 6.0)
    fig.tight_layout()
    fig.canvas.draw()  # applies the plot's fixed aspect, so its size is known
    plot_height = float(ax.get_position().height) * 6.0
    height = TOP_MARGIN + plot_height + DOTS_COLORBAR_OFFSET + DOTS_COLORBAR_HEIGHT + BOTTOM_MARGIN
    height = 2 * round(height * DPI / 2) / DPI  # a whole, even number of pixels
    fig.set_size_inches(FIGURE_WIDTH, height)
    fig.tight_layout()
    fig.canvas.draw()

    # Narrower than matplotlib leaves it, so the contour colorbar's tick labels and label fit beside it
    pos = ax.get_position()
    top = 1 - TOP_MARGIN / height
    scale = 0.92
    ax.set_position((pos.x0, top - pos.height * scale, pos.width * scale, pos.height * scale))
    pos = ax.get_position()
    bottom = top - pos.height
    contour_width = contour_colorbar.ax.get_position().width
    contour_colorbar.ax.set_position((pos.x1 + 0.02, bottom, contour_width, pos.height))
    dots_height = DOTS_COLORBAR_HEIGHT / height
    dots_colorbar.ax.set_position((pos.x0 + 0.1 * pos.width, bottom - DOTS_COLORBAR_OFFSET / height, 0.8 * pos.width, dots_height))


@dataclass
class Sweep:
    """What the animation reads from a 2d trajectory sweep file."""
    alphas: np.ndarray
    betas: np.ndarray
    loss: np.ndarray  # [len(betas), len(alphas)]
    acc: np.ndarray  # [len(betas), len(alphas)]
    trajectory: np.ndarray  # [n, 2], (alpha, beta) of every training snapshot on the plane
    explained_variance: np.ndarray | None
    run_id: str
    split: str
    max_step: int | None


@dataclass
class Scene:
    """The figure, built once, and the few artists that change from frame to frame."""
    fig: Figure
    ax: Axes
    filled: QuadContourSet
    artists: TrajectoryArtists
    label: Text
    trajectory: np.ndarray  # [n, 2]
    trajectory_loss: np.ndarray  # [n], what the dots are shaded by: the network's measured loss
    log_levels: bool
    log_error_levels: bool
    title: str
    loss_label: str
    metrics: SnapshotMetrics  # the network's measured loss and accuracy at each snapshot
    surfaces: SnapshotSurfaces  # the contour of each frame
    grids: np.ndarray  # [frames, len(betas), len(alphas)], the surfaces' loss or accuracy, whichever the contour shows
    counts: list[int]  # how many snapshots each frame shows
    level_values: np.ndarray  # what the contour levels and colors are worked out from: every frame's grid
    levels: np.ndarray | int
    norm: BoundaryNorm | None
    contour_artists: list[Any]  # the filled contour and its lines, replaced whenever the contour is redrawn
    drawn_values: np.ndarray  # the grid the contour is showing right now
    sweep: Sweep


def load_sweep(npz: Path) -> Sweep:
    if not npz.is_file():
        raise FileNotFoundError(f"sweep file not found: {npz}")
    with np.load(npz) as data:
        check_grid_sweep_schema_version(npz, data)
        mode = read_str(npz, data, "mode")
        if mode != "2d":
            raise ValueError(f"{npz} is a {mode} sweep. An animation needs a 2d sweep.")
        if "trajectory" not in data.files:
            raise ValueError(f"{npz} has no 'trajectory' key. Delete it and re-run to make one with --trajectory-plot.")
        return Sweep(
            alphas=data["alphas"],
            betas=data["betas"],
            loss=data["loss"],
            acc=data["acc"],
            trajectory=data["trajectory"],
            explained_variance=data["explained_variance"] if "explained_variance" in data.files else None,
            run_id=read_str(npz, data, "run_id"),
            split=read_str(npz, data, "split"),
            max_step=int(data["max_step"]) if "max_step" in data.files else None,
        )


def frame_counts(total: int, snapshots_per_frame: int) -> list[int]:
    """How many snapshots each frame shows: snapshots_per_frame, 2 * snapshots_per_frame, and so on, always ending on the whole run."""
    if snapshots_per_frame < 1:
        raise ValueError(f"--snapshots-per-frame must be at least 1, got {snapshots_per_frame}")
    counts = list(range(snapshots_per_frame, total + 1, snapshots_per_frame))
    if not counts or counts[-1] != total:
        counts.append(total)
    return counts


def build_scene(sweep: Sweep, value: str, metrics: SnapshotMetrics, surfaces: SnapshotSurfaces, counts: list[int]) -> Scene:
    """
    Draw everything that stays the same in every frame, once, then lay the figure out once.
    metrics: the network's measured loss and accuracy at each snapshot.
    surfaces: the contour of every frame, evaluated for the frames in counts.
    """
    if value not in ("loss", "acc"):
        raise ValueError(f"--value must be loss or acc, got {value!r}")
    if list(surfaces.frames) != list(counts):
        raise ValueError("the surfaces were evaluated for different frames than the animation draws")
    grids = surfaces.loss if value == "loss" else surfaces.acc
    shown = grids[-1]
    # One set of levels and colors that covers every frame, so the colors mean the same in all of them
    level_values = grids
    log_levels, log_error_levels = (True, False) if value == "loss" else (False, True)
    title = f"{split_label(sweep.split)} {'loss' if value == 'loss' else 'accuracy'}"
    if len(metrics.loss) != len(sweep.trajectory):
        raise ValueError(f"{len(metrics.loss)} measured snapshots for a path of {len(sweep.trajectory)} points")
    loss_label = f"{split_label(sweep.split)} loss of the network at that step"
    trajectory_loss = metrics.loss

    # The labels beside the plot are as tall as the plot, which is short, so they are broken over lines
    contour_label = f"{title} on the plane\nthrough the network\nat that step"
    fig, ax, filled = draw_contour_background(sweep.alphas, sweep.betas, shown, contour_label, log_levels=log_levels, log_error_levels=log_error_levels, level_values=level_values, equal_color_steps=True, figsize=(FIGURE_WIDTH, 6.0))
    contour_artists = list(ax.collections)  # the filled contour and its lines. The path is added after this
    levels, norm, _ = contour_levels(level_values, log_levels, log_error_levels, 20, equal_color_steps=True)
    # Anchors under the plot: the plot's own x coordinates, and a distance in inches below its bottom edge
    below_plot = offset_copy(ax.transAxes, fig=fig, x=0.0, y=-READOUT_OFFSET, units="inches")
    artists = draw_trajectory_overlay(
        fig, ax, sweep.trajectory, trajectory_loss, loss_label, sweep.explained_variance,
        log_dot_colors=True,
        legend_kwargs={"loc": "upper right", "bbox_to_anchor": (1.0, 0.0), "bbox_transform": below_plot, "ncol": 2, "frameon": False},
    )
    ax.set_aspect("equal")
    ax.set_ylabel(ax.get_ylabel().replace(", weight-space", ",\nweight-space"))
    ax.set_title(f"{title}\n{run_label(sweep.run_id, sweep.max_step)}")

    # The only thing that moves besides the dots and the contour: the step counter.
    label = ax.text(0.0, 0.0, "", transform=below_plot, va="top", ha="left", fontsize=11)

    # Freeze the scales: the dots' color range comes from the whole path, and the axes stop autoscaling
    artists.path.set_clim(float(trajectory_loss.min()), float(trajectory_loss.max()))
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.set_autoscale_on(False)
    dots_colorbar = artists.path.colorbar
    contour_colorbar = filled.colorbar
    assert dots_colorbar is not None and contour_colorbar is not None, "both colorbars were just drawn"
    layout_figure(fig, ax, contour_colorbar, dots_colorbar)  # once. A layout pass per frame could shift the axes by a pixel

    scene = Scene(
        fig=fig, ax=ax, filled=filled, artists=artists, label=label,
        trajectory=sweep.trajectory, trajectory_loss=trajectory_loss,
        log_levels=log_levels, log_error_levels=log_error_levels, title=title, loss_label=loss_label, metrics=metrics,
        surfaces=surfaces, grids=grids, counts=list(counts), level_values=level_values,
        levels=levels, norm=norm, contour_artists=contour_artists, drawn_values=shown, sweep=sweep,
    )
    set_frame(scene, len(sweep.trajectory))
    return scene


def frame_label(scene: Scene, n: int) -> str:
    """The text in the corner of frame n: the step, and the network's measured loss and accuracy at it."""
    data = split_label(scene.sweep.split)
    return f"step {n * SNAPSHOT_SPACING}\n{data} loss {scene.metrics.loss[n - 1]:.3f}\n{data} accuracy {scene.metrics.acc[n - 1]:.1%}"


def draw_contour(scene: Scene, values: np.ndarray) -> None:
    """Replace the contour with a grid's, keeping the levels and colors, which are worked out once for every frame."""
    ax = scene.ax
    zorders = [artist.get_zorder() for artist in scene.contour_artists]  # the filled contour, then its lines
    for artist in scene.contour_artists:
        artist.remove()
    filled = ax.contourf(scene.sweep.alphas, scene.sweep.betas, values, levels=scene.levels, norm=scene.norm)
    lines = ax.contour(scene.sweep.alphas, scene.sweep.betas, values, levels=scene.levels, colors="k", linewidths=0.4)
    # Lines drawn later than the star would land on top of it. Keep them just under it, as they are in a static plot.
    filled.set_zorder(zorders[0])
    lines.set_zorder(min(zorders[1], ax.lines[0].get_zorder() - 0.1))
    scene.contour_artists = [filled, lines]
    scene.filled = filled
    scene.drawn_values = values


def set_frame(scene: Scene, n: int) -> None:
    """Show the first n snapshots, and the contour of that frame. Everything else stays put."""
    if not 1 <= n <= len(scene.trajectory):
        raise ValueError(f"a frame shows between 1 and {len(scene.trajectory)} snapshots, got {n}")
    if n not in scene.counts:
        raise ValueError(f"no surface was evaluated for {n} snapshots. The frames are {scene.counts[:3]} and on to {scene.counts[-1]}")
    draw_contour(scene, scene.grids[scene.counts.index(n)])
    points = scene.trajectory[:n]
    scene.artists.path.set_offsets(points)
    scene.artists.path.set_array(scene.trajectory_loss[:n])
    scene.label.set_text(frame_label(scene, n))


def render_video(scene: Scene, counts: list[int], fps: int, out_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not on the PATH, so no video can be written")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(scene.fig, str(out_path), dpi=DPI):
        for n in counts:
            set_frame(scene, n)
            writer.grab_frame()


# ---- The preview: render the first, middle and last frames and check that they are drawn right ----


@dataclass
class FrameState:
    """Everything about one frame that has to match, or has to be right, read back from the figure objects."""
    n: int
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    levels: np.ndarray
    norm_limits: tuple[float | None, float | None]
    norm_boundaries: np.ndarray | None
    dots_clim: tuple[float, float]
    dots_colorbar_limits: tuple[float, float]
    offsets: np.ndarray
    dot_values: np.ndarray
    label: str
    start: tuple[float, float]
    end: tuple[float, float]
    star: tuple[float, float]
    pixel_size: tuple[float, float]
    contour_values: np.ndarray  # the grid the contour is showing


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def first_float(values: object) -> float:
    """The first number of a coordinate array, as a float. matplotlib's stubs type these as any array-like."""
    return float(np.asarray(values, dtype=float).ravel()[0])


def optional_float(value: object) -> float | None:
    return None if value is None else first_float(value)


def read_frame_state(scene: Scene, n: int) -> FrameState:
    set_frame(scene, n)
    ax, filled, path = scene.ax, scene.filled, scene.artists.path
    boundaries = getattr(filled.norm, "boundaries", None)
    colorbar = path.colorbar
    assert colorbar is not None, "the dots should have a colorbar"
    return FrameState(
        n=n,
        xlim=(float(ax.get_xlim()[0]), float(ax.get_xlim()[1])),
        ylim=(float(ax.get_ylim()[0]), float(ax.get_ylim()[1])),
        levels=np.array(filled.levels),
        norm_limits=(optional_float(filled.norm.vmin), optional_float(filled.norm.vmax)),
        norm_boundaries=None if boundaries is None else np.array(boundaries),
        dots_clim=(float(path.get_clim()[0]), float(path.get_clim()[1])),
        # The dots' colorbar is horizontal, so its value axis is x
        dots_colorbar_limits=(float(colorbar.ax.get_xlim()[0]), float(colorbar.ax.get_xlim()[1])),
        offsets=np.array(path.get_offsets()),
        dot_values=np.array(path.get_array()),
        label=scene.label.get_text(),
        start=(first_float(scene.artists.start.get_xdata()), first_float(scene.artists.start.get_ydata())),
        end=(first_float(scene.artists.end.get_xdata()), first_float(scene.artists.end.get_ydata())),
        star=(first_float(ax.lines[0].get_xdata()), first_float(ax.lines[0].get_ydata())),
        pixel_size=(float(scene.fig.get_size_inches()[0] * DPI), float(scene.fig.get_size_inches()[1] * DPI)),
        contour_values=np.array(scene.drawn_values),
    )


def check_states(states: list[FrameState], scene: Scene) -> list[Check]:
    """The checks that read the figure objects. Each frame is compared with the others and with values worked out from the sweep."""
    sweep = scene.sweep
    first = states[0]
    checks: list[Check] = []

    def all_equal(name: str, same: list[bool], detail: str) -> None:
        checks.append(Check(name, all(same), detail))

    expected_xlim = (float(sweep.alphas.min()), float(sweep.alphas.max()))
    expected_ylim = (float(sweep.betas.min()), float(sweep.betas.max()))
    all_equal("axis limits identical across frames", [s.xlim == first.xlim and s.ylim == first.ylim for s in states], f"x {first.xlim}, y {first.ylim}")
    checks.append(Check(
        "axis limits equal the grid's alpha and beta range",
        np.allclose(first.xlim, expected_xlim) and np.allclose(first.ylim, expected_ylim),
        f"expected x {expected_xlim}, y {expected_ylim}",
    ))

    expected_levels, _, _ = contour_levels(scene.level_values, scene.log_levels, scene.log_error_levels, 20, equal_color_steps=True)
    all_equal("contour levels identical across frames", [np.array_equal(s.levels, first.levels) for s in states], f"{len(first.levels)} levels")
    checks.append(Check(
        "contour levels equal the ones worked out from every frame's grid",
        isinstance(expected_levels, np.ndarray) and np.allclose(first.levels, expected_levels),
        f"{len(first.levels)} levels",
    ))
    all_equal(
        "contour color mapping identical across frames",
        [s.norm_limits == first.norm_limits and (
            (s.norm_boundaries is None and first.norm_boundaries is None)
            or (s.norm_boundaries is not None and first.norm_boundaries is not None and np.array_equal(s.norm_boundaries, first.norm_boundaries))
        ) for s in states],
        f"norm limits {first.norm_limits}",
    )

    expected_clim = (float(scene.trajectory_loss.min()), float(scene.trajectory_loss.max()))
    all_equal("dot color range identical across frames", [s.dots_clim == first.dots_clim for s in states], f"{first.dots_clim}")
    checks.append(Check(
        "dot color range equals the min and max loss over the whole path",
        np.allclose(first.dots_clim, expected_clim),
        f"expected {expected_clim}",
    ))
    all_equal("dots colorbar range equals the dot color range", [np.allclose(s.dots_colorbar_limits, s.dots_clim) for s in states], f"{first.dots_colorbar_limits}")

    all_equal(
        "frame f draws exactly the first n path points",
        [len(s.offsets) == s.n and np.array_equal(s.offsets, scene.trajectory[:s.n]) for s in states],
        "n = " + ", ".join(str(s.n) for s in states),
    )
    all_equal(
        "dot colors are the values the dots are shaded by, for the first n points",
        [np.allclose(s.dot_values, scene.trajectory_loss[:s.n]) for s in states],
        "",
    )
    all_equal(
        "the step label matches the frame",
        [s.label.splitlines()[0] == f"step {s.n * SNAPSHOT_SPACING}" for s in states],
        ", ".join(s.label.splitlines()[0] for s in states),
    )
    # Read the numbers back out of the text on the figure and compare them with the measured values for that step
    def printed(label: str) -> tuple[float, float]:
        loss = float(re.search(r"loss (\d+\.\d+)", label).group(1))  # pyright: ignore[reportOptionalMemberAccess]
        accuracy = float(re.search(r"accuracy (\d+\.\d+)%", label).group(1)) / 100  # pyright: ignore[reportOptionalMemberAccess]
        return loss, accuracy
    all_equal(
        "the loss and accuracy printed in each frame are the network's measured values at that step",
        [np.allclose(printed(s.label), (scene.metrics.loss[s.n - 1], scene.metrics.acc[s.n - 1]), atol=6e-4) for s in states],
        ", ".join(f"{printed(s.label)[0]:.3f} / {printed(s.label)[1]:.1%}" for s in states),
    )
    all_equal(
        "start marker is the first point and the end marker the last, in every frame",
        [s.start == (float(scene.trajectory[0, 0]), float(scene.trajectory[0, 1])) and s.end == (float(scene.trajectory[-1, 0]), float(scene.trajectory[-1, 1])) for s in states],
        "",
    )
    all_equal("the star for the final weights is at (0, 0) in every frame", [s.star == (0.0, 0.0) for s in states], "")
    grids, surfaces = scene.grids, scene.surfaces
    all_equal(
        "each frame draws its own surface",
        [np.array_equal(s.contour_values, grids[scene.counts.index(s.n)]) for s in states],
        "n = " + ", ".join(str(s.n) for s in states),
    )
    checks.append(Check(
        "the directions the surfaces lie along reproduce the sweep's saved path",
        surfaces.direction_error <= DIRECTION_TOLERANCE,
        f"largest difference {surfaces.direction_error:.2e}",
    ))
    # Frame f's network lies on frame f's surface, at its own dot, so the contour there is that network's loss
    last_difference = float(np.abs(surfaces.loss[-1] - sweep.loss).max())
    checks.append(Check(
        "the last frame's surface equals the sweep's saved grid, which is the plane through the final weights",
        last_difference <= 1e-3,
        f"largest difference {last_difference:.2e}",
    ))
    measured = scene.metrics.loss
    readings = np.array([sample_grid(sweep.alphas, sweep.betas, surfaces.loss[i], scene.trajectory[n - 1:n])[0] for i, n in enumerate(scene.counts)])
    errors = np.abs(readings - measured[np.array(scene.counts) - 1])
    worst = int(errors.argmax())
    checks.append(Check(
        "in every frame the contour at the latest path point reads the network's measured loss",
        bool(np.all(errors <= 0.02 + 0.1 * measured[np.array(scene.counts) - 1])),
        f"largest difference {errors[worst]:.3f} at snapshot {scene.counts[worst]}, where the loss is {measured[scene.counts[worst] - 1]:.3f}",
    ))
    all_equal(
        "figure pixel size identical across frames, and even",
        [s.pixel_size == first.pixel_size for s in states] and [int(first.pixel_size[0]) % 2 == 0 and int(first.pixel_size[1]) % 2 == 0],
        f"{first.pixel_size[0]:.0f} by {first.pixel_size[1]:.0f}",
    )
    return checks


def background_pixels(scene: Scene, n: int) -> np.ndarray:
    """Draw frame n with everything that depends on the frame or the path hidden, and return the pixels."""
    set_frame(scene, n)
    hidden = [scene.artists.path, scene.label, scene.artists.start, scene.artists.end, scene.artists.legend]
    for artist in hidden:
        artist.set_visible(False)
    canvas = scene.fig.canvas
    assert isinstance(canvas, FigureCanvasAgg), "the Agg backend is selected at the top of this module"
    canvas.draw()
    pixels = np.array(canvas.buffer_rgba())
    for artist in hidden:
        artist.set_visible(True)
    return pixels


def check_images(scene: Scene, first: int, last: int) -> list[Check]:
    """The checks on the pixels. The contour is redrawn each frame, so the background differs between frames. What must hold
    is that drawing a frame is a function of that frame alone."""
    a, b = background_pixels(scene, first), background_pixels(scene, last)
    again = background_pixels(scene, first)
    return [
        Check("drawing a frame again, after another frame, gives identical pixels, so nothing leaks between frames", bool(np.array_equal(a, again)), f"{a.shape[1]} by {a.shape[0]} pixels"),
        Check("the background differs between the first and last frames, so the contour is really redrawn", not bool(np.array_equal(a, b)), f"{int((a != b).any(axis=2).sum())} pixels differ"),
    ]


def run_preview(scene: Scene, counts: list[int], out_dir: Path, tags: str, stamp: str) -> bool:
    """Render the first, middle and last frames as stills, run every check, and print the results. True if all pass."""
    frame_numbers = [1, (len(counts) + 1) // 2, len(counts)]
    chosen = [counts[f - 1] for f in frame_numbers]
    out_dir.mkdir(parents=True, exist_ok=True)

    states = [read_frame_state(scene, n) for n in chosen]
    checks = check_states(states, scene)
    checks += check_images(scene, chosen[0], chosen[-1])

    for f, n in zip(frame_numbers, chosen):
        set_frame(scene, n)
        out_path = out_dir / viz_filename(stamp, tags, format_tags(preview=True, frame=f), ext="png")
        scene.fig.savefig(out_path, dpi=DPI)
        print(f"Saved preview frame {f} of {len(counts)} (step {n * SNAPSHOT_SPACING}) to {out_path}")

    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"{'PASS' if c.ok else 'FAIL'}  {c.name:<{width}}  {c.detail}")
    ok = all(c.ok for c in checks)
    print("All preview checks passed." if ok else "Some preview checks FAILED.")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate the training path on the loss surface of a full-run trajectory sweep, as an MP4")
    parser.add_argument("run_id", type=str, help="run id of a training run, i.e. the name of its folder under checkpoints/, e.g. 2026-10-14T19:28:51_45xyzktb")
    parser.add_argument("--max-step", type=int, default=None, help="use only the training snapshots up to this step (e.g. 600), as plot's --max-step does. Default is the whole run")
    parser.add_argument("--value", choices=["loss", "acc"], default="loss", help="which metric this animation tracks")
    parser.add_argument("--snapshots-per-frame", type=int, default=5, help="how many snapshots progress per frame of video")
    parser.add_argument("--fps", type=int, default=20, help="output video fps")
    parser.add_argument("--preview", action="store_true", help="render only the first, middle and last frames as stills")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The 2d trajectory sweep is computed first if it isn't saved yet, which takes a while
    npz = create_or_load_grid_sweep(args.run_id, "2d", trajectory=True, max_step=args.max_step)
    sweep = load_sweep(npz)
    counts = frame_counts(len(sweep.trajectory), args.snapshots_per_frame)
    metrics = create_or_load_snapshot_metrics(sweep.run_id, sweep.split, len(sweep.trajectory))
    surfaces = create_or_load_snapshot_surfaces(sweep.run_id, sweep.split, sweep.trajectory, sweep.alphas, sweep.betas, counts)
    scene = build_scene(sweep, args.value, metrics, surfaces, counts)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = viz_animations_dir(sweep.run_id)
    tags = f"{sweep_tags(npz)}_{format_tags(value=args.value, snapshots_per_frame=args.snapshots_per_frame)}"

    if args.preview:
        sys.exit(0 if run_preview(scene, counts, out_dir, tags, stamp) else 1)

    out_path = out_dir / viz_filename(stamp, tags, format_tags(fps=args.fps), ext="mp4")
    render_video(scene, counts, args.fps, out_path)
    print(f"Saved {len(counts)} frames ({counts[0] * SNAPSHOT_SPACING} to {counts[-1] * SNAPSHOT_SPACING} steps, {len(counts) / args.fps:.1f} seconds at {args.fps} fps) to {out_path}")


if __name__ == "__main__":
    main()
