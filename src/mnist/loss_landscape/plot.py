# matplotlib ships partial type information, so strict mode flags its calls as partially unknown.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_1d(alphas: np.ndarray, losses: np.ndarray, accs: np.ndarray, out_path: Path) -> None:
    """
    Plot validation loss and accuracy along one direction through the trained weights.
    alphas: [N] step sizes along the direction (0 is the trained model)
    losses: [N] loss at each alpha
    accs:   [N] accuracy at each alpha
    """
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

    ax_loss.plot(alphas, losses, marker="o")
    ax_loss.set_ylabel("validation loss")
    ax_loss.axvline(0, color="gray", linestyle="--", linewidth=1)  # θ*, the trained model

    ax_acc.plot(alphas, accs, marker="o", color="tab:orange")
    ax_acc.set_ylabel("validation accuracy")
    ax_acc.set_xlabel("alpha (step along d1, relative to filter norm)")
    ax_acc.axvline(0, color="gray", linestyle="--", linewidth=1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_2d_contour(alphas: np.ndarray, betas: np.ndarray, values: np.ndarray, out_path: Path, title: str, log_levels: bool = True, n_levels: int = 20, trajectory: np.ndarray | None = None) -> None:
    """
    Filled contour plot of a [len(betas), len(alphas)] grid, with the trained weights (alpha = beta = 0) marked.
    log_levels spaces the contour levels geometrically, which suits a loss that grows over orders of magnitude.
    trajectory: optional [n, 2] array of (alpha, beta) points, the training path, drawn on top of the contour.
    """
    levels = np.geomspace(max(float(values.min()), 1e-6), float(values.max()), n_levels) if log_levels else n_levels

    fig, ax = plt.subplots(figsize=(7, 6))
    filled = ax.contourf(alphas, betas, values, levels=levels)
    ax.contour(alphas, betas, values, levels=levels, colors="k", linewidths=0.4)
    fig.colorbar(filled, ax=ax, label=title)
    ax.plot(0, 0, marker="*", color="white", markeredgecolor="k", markersize=14)  # θ*, the trained model
    if trajectory is None:
        ax.set_xlabel("alpha (step along d1, relative to filter norm)")
        ax.set_ylabel("beta (step along d2, relative to filter norm)")
    else:
        # PCA directions are unit vectors, so the axes are plain weight-space distance
        ax.set_xlabel("alpha (PC1, weight-space distance)")
        ax.set_ylabel("beta (PC2, weight-space distance)")
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="white", linewidth=1, zorder=3)
        ax.scatter(trajectory[:, 0], trajectory[:, 1], color="red", s=8, zorder=4)
        ax.plot(trajectory[0, 0], trajectory[0, 1], marker="o", color="lime", markeredgecolor="k", markersize=9, zorder=5, label="start")
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], marker="s", color="cyan", markeredgecolor="k", markersize=8, zorder=5, label="end")
        ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.set_title(title)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_2d_surface(alphas: np.ndarray, betas: np.ndarray, values: np.ndarray, out_path: Path, title: str) -> None:
    """
    3D surface plot of a [len(betas), len(alphas)] grid.
    """
    alpha_mesh, beta_mesh = np.meshgrid(alphas, betas)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(alpha_mesh, beta_mesh, values, cmap="viridis")
    ax.set_xlabel("alpha (d1)")
    ax.set_ylabel("beta (d2)")
    ax.set_zlabel(title)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a plot from an .npz saved by compute_surface")
    parser.add_argument("npz", type=Path, help="path to a sweep; the _1d / _2d suffix of the name picks the plot, e.g. plots/abc123_2d.npz")
    parser.add_argument("--style", choices=["contour", "surface"], default="contour", help="2D plots only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.npz.is_file():
        raise FileNotFoundError(f"sweep file not found: {args.npz}")
    if args.npz.stem.endswith("_1d"):
        with np.load(args.npz) as data:
            alphas, losses, accs = data["alphas"], data["loss"], data["acc"]
        out_path = args.npz.with_suffix(".png")
        plot_1d(alphas, losses, accs, out_path)
        print("Saved 1D plot to", out_path)
    elif args.npz.stem.endswith("_2d"):
        with np.load(args.npz) as data:
            alphas, betas, losses, accs = data["alphas"], data["betas"], data["loss"], data["acc"]
            # Only present for sweeps run with --trajectory-plot
            trajectory = data["trajectory"] if "trajectory" in data.files else None
        for name, values, log_levels in (("loss", losses, True), ("acc", accs, False)):
            out_path = args.npz.with_name(f"{args.npz.stem}_{name}_{args.style}.png")
            title = f"validation {'loss' if name == 'loss' else 'accuracy'}"
            if args.style == "contour":
                plot_2d_contour(alphas, betas, values, out_path, title, log_levels=log_levels, trajectory=trajectory)
            else:
                plot_2d_surface(alphas, betas, values, out_path, title)
            print(f"Saved 2D {args.style} plot to", out_path)
    else:
        raise ValueError(f"expected a file name ending in _1d or _2d, got {args.npz.name}")


if __name__ == "__main__":
    main()
