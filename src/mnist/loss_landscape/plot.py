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
