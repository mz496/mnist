# matplotlib ships partial type information, so strict mode flags its calls as partially unknown.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, LogNorm
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

from mnist.loss_landscape.grid_sweep_file_manager import check_grid_sweep_schema_version, read_str


def run_label(run_id: str, max_step: int | None) -> str:
    """The run line of a title, naming the window of training steps when the sweep used only the first ones."""
    return f"run {run_id}" if max_step is None else f"run {run_id}, first {max_step} steps"


def split_label(split: str) -> str:
    """The name of the data a sweep was measured on ("train" or "validation"), as it should appear in titles."""
    return "training" if split == "train" else split


def plot_1d(alphas: np.ndarray, losses: np.ndarray, accs: np.ndarray, out_path: Path, run_id: str | None = None, split: str = "validation") -> None:
    """
    Plot loss and accuracy along one direction through the trained weights.
    split: which data the sweep was measured on, used in the axis labels.
    alphas: [N] step sizes along the direction (0 is the trained model)
    losses: [N] loss at each alpha
    accs:   [N] accuracy at each alpha
    run_id: shown in the title, if given
    """
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    if run_id is not None:
        fig.suptitle(f"run {run_id}")

    ax_loss.plot(alphas, losses, marker="o")
    ax_loss.set_ylabel(f"{split_label(split)} loss")
    ax_loss.axvline(0, color="gray", linestyle="--", linewidth=1)  # θ*, the trained model

    ax_acc.plot(alphas, accs, marker="o", color="tab:orange")
    ax_acc.set_ylabel(f"{split_label(split)} accuracy")
    ax_acc.set_xlabel("alpha (step along d1, relative to filter norm)")
    ax_acc.axvline(0, color="gray", linestyle="--", linewidth=1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def sample_grid(alphas: np.ndarray, betas: np.ndarray, grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Bilinearly interpolate a [len(betas), len(alphas)] grid at [n, 2] points of (alpha, beta). Returns [n].
    Points outside the grid are clamped to its edge cells.
    """
    i = np.clip(np.searchsorted(alphas, points[:, 0]) - 1, 0, len(alphas) - 2)
    j = np.clip(np.searchsorted(betas, points[:, 1]) - 1, 0, len(betas) - 2)
    ta = (points[:, 0] - alphas[i]) / (alphas[i + 1] - alphas[i])
    tb = (points[:, 1] - betas[j]) / (betas[j + 1] - betas[j])
    return (
        grid[j, i] * (1 - ta) * (1 - tb)
        + grid[j, i + 1] * ta * (1 - tb)
        + grid[j + 1, i] * (1 - ta) * tb
        + grid[j + 1, i + 1] * ta * tb
    )


def contour_levels(values: np.ndarray, log_levels: bool, log_error_levels: bool, n_levels: int, equal_color_steps: bool = False) -> tuple[np.ndarray | int, BoundaryNorm | None, bool]:
    """
    The contour levels for a grid, the color norm to go with them, and whether the grid is flat.
    log_levels spaces the levels geometrically, which suits a loss that grows over orders of magnitude.
    log_error_levels is for values that approach 1 (accuracy): the levels are spaced geometrically in (1 - value),
    so they crowd toward 1.0, where a plain linear or log scale on the value itself would look flat.
    equal_color_steps gives every level band the same step of the colormap, which with log-spaced levels makes the colors
    a log scale. Without it the colors follow the value linearly, so a loss that is mostly small is mostly one dark color.
    """
    levels: np.ndarray | int
    # By default the colors follow the value linearly, so levels crowded near 1.0 would all land on one end of the colormap.
    # A BoundaryNorm gives each band its own equal step of the colormap instead.
    norm: BoundaryNorm | None = None
    # A grid with a single value, such as a plateau where every point in a small window predicts the same, has no spread
    # to space contour levels over. Draw it as one band.
    flat = float(values.max() - values.min()) == 0.0
    if flat:
        levels = np.array([float(values.min()) - 5e-4, float(values.max()) + 5e-4])
    elif log_error_levels:
        errors = np.geomspace(max(1.0 - float(values.max()), 1e-6), 1.0 - float(values.min()), n_levels)
        levels = 1.0 - errors[::-1]  # reversed so the levels increase
        norm = BoundaryNorm(levels, ncolors=256)
    elif log_levels:
        levels = np.geomspace(max(float(values.min()), 1e-6), float(values.max()), n_levels)
        if equal_color_steps:
            norm = BoundaryNorm(levels, ncolors=256)
    else:
        levels = n_levels
    return levels, norm, flat


def draw_contour_background(alphas: np.ndarray, betas: np.ndarray, values: np.ndarray, title: str, log_levels: bool = True, n_levels: int = 20, log_error_levels: bool = False, level_values: np.ndarray | None = None, equal_color_steps: bool = False, figsize: tuple[float, float] = (7, 6)) -> tuple[Figure, Axes, QuadContourSet]:
    """
    Make the figure and draw the filled contour of a [len(betas), len(alphas)] grid with its colorbar and the star for the
    trained weights (alpha = beta = 0). Everything the path is drawn on top of, and nothing that depends on the path.
    level_values: what the contour levels and colors are worked out from, when they should cover more than this one grid.
    Pass several grids here to give them all one shared scale.
    equal_color_steps: see contour_levels. figsize: in inches, for a figure that needs room around the plot.
    """
    levels, norm, flat = contour_levels(values if level_values is None else level_values, log_levels, log_error_levels, n_levels, equal_color_steps)
    fig, ax = plt.subplots(figsize=figsize)
    filled = ax.contourf(alphas, betas, values, levels=levels, norm=norm)
    if not flat:
        ax.contour(alphas, betas, values, levels=levels, colors="k", linewidths=0.4)
    fig.colorbar(filled, ax=ax, label=title)
    ax.plot(0, 0, marker="*", color="white", markeredgecolor="k", markersize=14)  # θ*, the trained model
    return fig, ax, filled


@dataclass
class TrajectoryArtists:
    """The pieces drawn for a training path, kept so a caller can update them."""
    path: PathCollection
    start: Line2D
    end: Line2D
    legend: Legend


def draw_trajectory_overlay(fig: Figure, ax: Axes, trajectory: np.ndarray, trajectory_loss: np.ndarray | None, trajectory_loss_label: str, explained_variance: np.ndarray | None, log_dot_colors: bool = False, colorbar_pad: float = 0.15, legend_kwargs: dict[str, Any] | None = None) -> TrajectoryArtists:
    """
    Label the axes for the PCA plane and draw the training path on a contour background: the dots, a colorbar for them
    when they are shaded by loss, the start and end markers, and the legend.
    trajectory: [n, 2] array of (alpha, beta) points.
    trajectory_loss: optional [n] loss at each point. The dots fade linearly from white at the smallest to red at the largest.
    explained_variance: optional [2] fraction of the snapshots' variance along PC1 and PC2, shown in the axis labels.
    log_dot_colors: fade the dots by the log of the loss instead, so a loss that is mostly small is not mostly white.
    colorbar_pad: the gap under the plot before the dots' colorbar, as a fraction of the plot's height.
    legend_kwargs: where to put the legend, as keyword arguments for Axes.legend. By default it's inside, at the upper right.
    """
    # PCA directions are unit vectors, so the axes are plain weight-space distance
    x_label, y_label = "alpha (PC1, weight-space distance)", "beta (PC2, weight-space distance)"
    if explained_variance is not None:
        x_label = f"alpha (PC1, {float(explained_variance[0]):.1%} of variance, weight-space distance)"
        y_label = f"beta (PC2, {float(explained_variance[1]):.1%} of variance, weight-space distance)"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if trajectory_loss is None:
        path = ax.scatter(trajectory[:, 0], trajectory[:, 1], color="red", s=2, zorder=4)
    else:
        # Fade linearly with loss: full red at the largest loss on the path, full white at the smallest
        white_to_red = LinearSegmentedColormap.from_list("white_to_red", ["white", "red"])
        dot_norm = LogNorm(vmin=float(trajectory_loss.min()), vmax=float(trajectory_loss.max())) if log_dot_colors else None
        path = ax.scatter(trajectory[:, 0], trajectory[:, 1], c=trajectory_loss, cmap=white_to_red, norm=dot_norm, s=2, linewidths=0, zorder=4)
        # Horizontal, under the plot, so it doesn't squeeze the axes next to the other colorbar
        dots_colorbar = fig.colorbar(path, ax=ax, label=trajectory_loss_label, orientation="horizontal", location="bottom", pad=colorbar_pad, shrink=0.8)
        if log_dot_colors:
            # A log axis would label only the powers of ten. Name a few round values inside the range instead.
            ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0) if float(trajectory_loss.min()) <= t <= float(trajectory_loss.max())]
            dots_colorbar.set_ticks(ticks)
            dots_colorbar.set_ticklabels([f"{t:g}" for t in ticks])
            dots_colorbar.ax.minorticks_off()
    start = ax.plot(trajectory[0, 0], trajectory[0, 1], marker="o", color="lime", markeredgecolor="k", markersize=9, zorder=5, label="start")[0]
    end = ax.plot(trajectory[-1, 0], trajectory[-1, 1], marker="s", color="cyan", markeredgecolor="k", markersize=8, zorder=5, label="end")[0]
    legend = ax.legend(**legend_kwargs) if legend_kwargs is not None else ax.legend(loc="upper right")
    return TrajectoryArtists(path=path, start=start, end=end, legend=legend)


def plot_2d_contour(alphas: np.ndarray, betas: np.ndarray, values: np.ndarray, out_path: Path, title: str, log_levels: bool = True, n_levels: int = 20, trajectory: np.ndarray | None = None, log_error_levels: bool = False, run_id: str | None = None, explained_variance: np.ndarray | None = None, trajectory_loss: np.ndarray | None = None, trajectory_loss_label: str = "loss", max_step: int | None = None) -> None:
    """
    Filled contour plot of a [len(betas), len(alphas)] grid, with the trained weights (alpha = beta = 0) marked.
    See contour_levels for log_levels and log_error_levels.
    trajectory: optional [n, 2] array of (alpha, beta) points, the training path, drawn on top of the contour.
    run_id: shown in the title, if given.
    max_step: the last training step of the window the sweep used, shown in the title next to the run id.
    explained_variance: optional [2] fraction of the snapshots' variance along PC1 and PC2, shown in the axis labels.
    """
    fig, ax, _ = draw_contour_background(alphas, betas, values, title, log_levels, n_levels, log_error_levels)
    if trajectory is None:
        ax.set_xlabel("alpha (step along d1, relative to filter norm)")
        ax.set_ylabel("beta (step along d2, relative to filter norm)")
    else:
        draw_trajectory_overlay(fig, ax, trajectory, trajectory_loss, trajectory_loss_label, explained_variance)
    ax.set_aspect("equal")
    ax.set_title(title if run_id is None else f"{title}\n{run_label(run_id, max_step)}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_2d_surface(alphas: np.ndarray, betas: np.ndarray, values: np.ndarray, out_path: Path, title: str, run_id: str | None = None, max_step: int | None = None) -> None:
    """
    3D surface plot of a [len(betas), len(alphas)] grid.
    run_id: shown in the title, if given.
    max_step: the last training step of the window the sweep used, shown in the title next to the run id.
    """
    alpha_mesh, beta_mesh = np.meshgrid(alphas, betas)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(alpha_mesh, beta_mesh, values, cmap="viridis")
    ax.set_xlabel("alpha (d1)")
    ax.set_ylabel("beta (d2)")
    ax.set_zlabel(title)
    if run_id is not None:
        ax.set_title(f"{title}\n{run_label(run_id, max_step)}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a plot from an .npz saved by compute_surface")
    parser.add_argument("npz", type=Path, help="path to a sweep saved by compute_surface, e.g. plots/abc123_2d.npz. The file's mode key says whether it is a 1d or a 2d sweep")
    parser.add_argument("--style", choices=["contour", "surface"], default="contour", help="2D plots only")
    return parser.parse_args()


def draw_1d(npz: Path, data: np.lib.npyio.NpzFile, stamp: str) -> None:
    alphas, losses, accs = data["alphas"], data["loss"], data["acc"]
    run_id = read_str(npz, data, "run_id")
    split = read_str(npz, data, "split")
    out_path = npz.with_name(f"{npz.stem}_{stamp}.png")
    plot_1d(alphas, losses, accs, out_path, run_id=run_id, split=split)
    print("Saved 1D plot to", out_path)


def draw_2d(npz: Path, data: np.lib.npyio.NpzFile, style: str, stamp: str) -> None:
    alphas, betas, losses, accs = data["alphas"], data["betas"], data["loss"], data["acc"]
    # Only present for sweeps run with --trajectory-plot
    trajectory = data["trajectory"] if "trajectory" in data.files else None
    explained_variance = data["explained_variance"] if "explained_variance" in data.files else None
    run_id = read_str(npz, data, "run_id")
    split = read_str(npz, data, "split")
    # Only present for sweeps run with --max-step
    max_step = int(data["max_step"]) if "max_step" in data.files else None
    # The loss at each path point, read off the saved loss grid at that point's position. Not recomputed per snapshot.
    trajectory_loss = sample_grid(alphas, betas, losses, trajectory) if trajectory is not None else None
    for name, values, log_levels, log_error_levels in (("loss", losses, True, False), ("acc", accs, False, True)):
        out_path = npz.with_name(f"{npz.stem}_{name}_{style}_{stamp}.png")
        title = f"{split_label(split)} {'loss' if name == 'loss' else 'accuracy'}"
        if style == "contour":
            plot_2d_contour(alphas, betas, values, out_path, title, log_levels=log_levels, trajectory=trajectory, log_error_levels=log_error_levels, run_id=run_id, explained_variance=explained_variance, trajectory_loss=trajectory_loss, trajectory_loss_label=f"{split_label(split)} loss at path point (from the grid)", max_step=max_step)
        else:
            plot_2d_surface(alphas, betas, values, out_path, title, run_id=run_id, max_step=max_step)
        print(f"Saved 2D {style} plot to", out_path)


def main() -> None:
    args = parse_args()
    if not args.npz.is_file():
        raise FileNotFoundError(f"sweep file not found: {args.npz}")
    with np.load(args.npz) as data:
        # Before anything else: the other keys can't be trusted from a file this reader doesn't understand
        check_grid_sweep_schema_version(args.npz, data)
        mode = read_str(args.npz, data, "mode")
        # One timestamp for every image this run writes, so a re-plot never overwrites an earlier one
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if mode == "1d":
            draw_1d(args.npz, data, stamp)
        elif mode == "2d":
            draw_2d(args.npz, data, args.style, stamp)
        else:
            raise ValueError(f"unknown mode {mode!r} in {args.npz}, expected '1d' or '2d'")


if __name__ == "__main__":
    main()
