
import argparse
import copy
from pathlib import Path

import torch
import numpy as np

from mnist.config import load_default_config
from mnist.data import get_dataloaders
from mnist.evaluate import evaluate
from mnist.loss_landscape.lib import create_random_direction
from mnist.loss_landscape.plot import plot_1d
from mnist.model import LeNet5
from mnist.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test set")
    parser.add_argument("checkpoint", type=Path, help="path to a .pt state_dict, e.g. checkpoints/abc123.pt")
    parser.add_argument("--seed", type=int, default=0, help="seed for the random directions")
    return parser.parse_args()

def load_checkpoint(path: Path, device: str) -> LeNet5:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    model = LeNet5().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    device = get_device()
    print("MOVING MODEL TO DEVICE:", device)
    # Fetch saved weights from the checkpoint (theta star)
    model = load_checkpoint(checkpoint_path, device)
    theta_star = [p.detach().clone() for p in model.parameters()]
    model_shifted = copy.deepcopy(model)

    config = load_default_config()
    data = get_dataloaders(config.data)

    # Get two random directions in weight-space of the model
    # These will be our axes for plotting
    print("DIRECTION SEED:", args.seed)
    d1 = create_random_direction(model, args.seed)

    # 1D sweep first
    # Evaluate as if weights were shifted
    x_min = -1
    x_max = 1
    num_points = 21
    alphas = np.linspace(x_min, x_max, num_points)

    losses: list[float] = []
    accs: list[float] = []
    for alpha in alphas:
        with torch.no_grad():
            for p, w, d in zip(model_shifted.parameters(), theta_star, d1):
                # Shift the model alpha steps in the direction d1
                p.copy_(w + float(alpha) * d)
            # Evaluate at that point
            loss, acc = evaluate(model_shifted, data.validation, device)
            losses.append(loss)
            accs.append(acc)

    out_path = Path("plots") / f"{checkpoint_path.stem}_1d.png"
    plot_1d(alphas, np.array(losses), np.array(accs), out_path)
    print("Saved 1D plot to", out_path)

    # TODO 10-14: 2D sweep.
    # In a grid from [-x,+x]*[-y,+y] eval as if the weights were shifted to that space
    # Plot the surface

if __name__ == "__main__":
    main()