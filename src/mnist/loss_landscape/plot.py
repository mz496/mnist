# matplotlib ships partial type information, so strict mode flags its calls as partially unknown.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap


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


def plot_2d_contour(alphas: np.ndarray, betas: np.ndarray, values: np.ndarray, out_path: Path, title: str, log_levels: bool = True, n_levels: int = 20, trajectory: np.ndarray | None = None, log_error_levels: bool = False, run_id: str | None = None, explained_variance: np.ndarray | None = None, trajectory_loss: np.ndarray | None = None, trajectory_loss_label: str = "loss") -> None:
    """
    Filled contour plot of a [len(betas), len(alphas)] grid, with the trained weights (alpha = beta = 0) marked.
    log_levels spaces the contour levels geometrically, which suits a loss that grows over orders of magnitude.
    log_error_levels is for values that approach 1 (accuracy): the levels are spaced geometrically in (1 - value),
    so they crowd toward 1.0, where a plain linear or log scale on the value itself would look flat.
    trajectory: optional [n, 2] array of (alpha, beta) points, the training path, drawn on top of the contour.
    run_id: shown in the title, if given.
    explained_variance: optional [2] fraction of the snapshots' variance along PC1 and PC2, shown in the axis labels.
    """
    levels: np.ndarray | int
    # By default the colors follow the value linearly, so levels crowded near 1.0 would all land on one end of the colormap.
    # A BoundaryNorm gives each band its own equal step of the colormap instead.
    norm: BoundaryNorm | None = None
    if log_error_levels:
        errors = np.geomspace(max(1.0 - float(values.max()), 1e-6), 1.0 - float(values.min()), n_levels)
        levels = 1.0 - errors[::-1]  # reversed so the levels increase
        norm = BoundaryNorm(levels, ncolors=256)
    elif log_levels:
        levels = np.geomspace(max(float(values.min()), 1e-6), float(values.max()), n_levels)
    else:
        levels = n_levels

    fig, ax = plt.subplots(figsize=(7, 6))
    filled = ax.contourf(alphas, betas, values, levels=levels, norm=norm)
    ax.contour(alphas, betas, values, levels=levels, colors="k", linewidths=0.4)
    fig.colorbar(filled, ax=ax, label=title)
    ax.plot(0, 0, marker="*", color="white", markeredgecolor="k", markersize=14)  # θ*, the trained model
    if trajectory is None:
        ax.set_xlabel("alpha (step along d1, relative to filter norm)")
        ax.set_ylabel("beta (step along d2, relative to filter norm)")
    else:
        # PCA directions are unit vectors, so the axes are plain weight-space distance
        x_label, y_label = "alpha (PC1, weight-space distance)", "beta (PC2, weight-space distance)"
        if explained_variance is not None:
            x_label = f"alpha (PC1, {float(explained_variance[0]):.1%} of variance, weight-space distance)"
            y_label = f"beta (PC2, {float(explained_variance[1]):.1%} of variance, weight-space distance)"
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if trajectory_loss is None:
            ax.scatter(trajectory[:, 0], trajectory[:, 1], color="red", s=2, zorder=4)
        else:
            # Fade linearly with loss: full red at the largest loss on the path, full white at the smallest
            white_to_red = LinearSegmentedColormap.from_list("white_to_red", ["white", "red"])
            path_points = ax.scatter(trajectory[:, 0], trajectory[:, 1], c=trajectory_loss, cmap=white_to_red, s=2, linewidths=0, zorder=4)
            # Horizontal, under the plot, so it doesn't squeeze the axes next to the other colorbar
            fig.colorbar(path_points, ax=ax, label=trajectory_loss_label, orientation="horizontal", location="bottom", pad=0.15, shrink=0.8)
        ax.plot(trajectory[0, 0], trajectory[0, 1], marker="o", color="lime", markeredgecolor="k", markersize=9, zorder=5, label="start")
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], marker="s", color="cyan", markeredgecolor="k", markersize=8, zorder=5, label="end")
        ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.set_title(title if run_id is None else f"{title}\nrun {run_id}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_2d_surface(alphas: np.ndarray, betas: np.ndarray, values: np.ndarray, out_path: Path, title: str, run_id: str | None = None) -> None:
    """
    3D surface plot of a [len(betas), len(alphas)] grid.
    run_id: shown in the title, if given.
    """
    alpha_mesh, beta_mesh = np.meshgrid(alphas, betas)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(alpha_mesh, beta_mesh, values, cmap="viridis")
    ax.set_xlabel("alpha (d1)")
    ax.set_ylabel("beta (d2)")
    ax.set_zlabel(title)
    if run_id is not None:
        ax.set_title(f"{title}\nrun {run_id}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a plot from an .npz saved by compute_surface")
    parser.add_argument("npz", type=Path, help="path to a sweep saved by compute_surface, e.g. plots/abc123_2d.npz. The file's mode key says whether it is a 1d or a 2d sweep")
    parser.add_argument("--style", choices=["contour", "surface"], default="contour", help="2D plots only")
    return parser.parse_args()


def draw_1d(npz: Path, data: np.lib.npyio.NpzFile) -> None:
    alphas, losses, accs = data["alphas"], data["loss"], data["acc"]
    run_id = str(data["run_id"]) if "run_id" in data.files else None
    # Older files don't record the split; they were all measured on validation
    split = str(data["split"]) if "split" in data.files else "validation"
    out_path = npz.with_suffix(".png")
    plot_1d(alphas, losses, accs, out_path, run_id=run_id, split=split)
    print("Saved 1D plot to", out_path)


def draw_2d(npz: Path, data: np.lib.npyio.NpzFile, style: str) -> None:
    alphas, betas, losses, accs = data["alphas"], data["betas"], data["loss"], data["acc"]
    # Only present for sweeps run with --trajectory-plot
    trajectory = data["trajectory"] if "trajectory" in data.files else None
    explained_variance = data["explained_variance"] if "explained_variance" in data.files else None
    run_id = str(data["run_id"]) if "run_id" in data.files else None
    # Older files don't record the split; they were all measured on validation
    split = str(data["split"]) if "split" in data.files else "validation"
    # The loss at each path point, read off the saved loss grid at that point's position. Not recomputed per snapshot.
    trajectory_loss = sample_grid(alphas, betas, losses, trajectory) if trajectory is not None else None
    for name, values, log_levels, log_error_levels in (("loss", losses, True, False), ("acc", accs, False, True)):
        out_path = npz.with_name(f"{npz.stem}_{name}_{style}.png")
        title = f"{split_label(split)} {'loss' if name == 'loss' else 'accuracy'}"
        if style == "contour":
            plot_2d_contour(alphas, betas, values, out_path, title, log_levels=log_levels, trajectory=trajectory, log_error_levels=log_error_levels, run_id=run_id, explained_variance=explained_variance, trajectory_loss=trajectory_loss, trajectory_loss_label=f"{split_label(split)} loss at path point (from the grid)")
        else:
            plot_2d_surface(alphas, betas, values, out_path, title, run_id=run_id)
        print(f"Saved 2D {style} plot to", out_path)


def main() -> None:
    args = parse_args()
    if not args.npz.is_file():
        raise FileNotFoundError(f"sweep file not found: {args.npz}")
    with np.load(args.npz) as data:
        if "mode" not in data.files:
            raise ValueError(f"{args.npz} has no 'mode' key, so it can't be told apart as a 1d or 2d sweep. Regenerate it with the current compute_surface.")
        mode = str(data["mode"])
        if mode == "1d":
            draw_1d(args.npz, data)
        elif mode == "2d":
            draw_2d(args.npz, data, args.style)
        else:
            raise ValueError(f"unknown mode {mode!r} in {args.npz}, expected '1d' or '2d'")


if __name__ == "__main__":
    main()
