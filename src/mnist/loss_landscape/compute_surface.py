
import argparse
import copy
from pathlib import Path

import torch
import numpy as np

from mnist.config import load_default_config
from mnist.data import get_dataloaders
from mnist.evaluate import evaluate
from mnist.loss_landscape.lib import create_random_direction
from mnist.model import LeNet5
from mnist.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep a checkpoint's weights along a random direction, evaluate the validation loss and accuracy at each point, and save the results to an .npz in plots/"
    )
    parser.add_argument("checkpoint", type=Path, help="path to a .pt state_dict, e.g. checkpoints/abc123.pt")
    parser.add_argument("--trajectory-path", type=Path, help="turn on non-random directions mode so that we can plot a trajectory on the plot - pass in a subdir for the trajectory snapshots, e.g. checkpoints/snapshots-abc123")
    parser.add_argument("--seed", type=int, default=0, help="when in random direction mode, seed for the randomness (2d mode also uses seed + 1)")
    parser.add_argument("--mode", choices=["1d", "2d"], default="1d", help="sweep along one direction or over a 2D grid")
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

    if args.mode == "1d":
        if args.trajectory_path is not None:
            raise Exception("trajectory plot not supported for 1D")

        # Get a random direction in weight-space of the model
        # This will be our axis for plotting
        print("DIRECTION SEED:", args.seed)
        d1 = create_random_direction(model, args.seed)

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

        # Save the raw numbers
        out_path = Path("plots") / f"{checkpoint_path.stem}_1d.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            alphas=alphas,
            loss=np.array(losses),
            acc=np.array(accs),
            seed=np.array(args.seed),
            checkpoint=np.array(str(checkpoint_path)),
        )
        print("Saved 1D sweep to", out_path)

    elif args.mode == "2d":
        if args.trajectory_path is not None:
            # Use training snapshots as the plotting plane and to plot the trajectory on that plane

            # TODO[trajectory-plotting] Load training snapshots for run id

            # TODO[trajectory-plotting] PCA on training snapshots; extract top 2
            # Set d1, d2 here
            # Do explained-variance check
            # Fix signs? TO ma Something like:
            # def fix_signs(components: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # """
            # components: [2, D] the principal directions
            # coords: [n, 2] each snapshot's coordinates along them
            # Flip each component so the snapshot with the largest |coordinate| on it is positive.
            # """
            # for k in range(components.shape[0]):
            #     ref = int(np.argmax(np.abs(coords[:, k])))
            #     if coords[ref, k] < 0:
            #         components[k] *= -1
            #         coords[:, k] *= -1
            # return components, coords

            # TODO[trajectory-plotting] Project each snapshot onto PCAs to get (alpha_s, beta_s) per-snapshot points
            # Set xmin, xmax, ymin, ymax here

            # TODO[trajectory-plotting] Save the file
            pass
        else:
            # Get two random directions in weight-space of the model
            # These will be our axes for plotting
            print("DIRECTION SEEDS:", args.seed, args.seed + 1)
            d1 = create_random_direction(model, args.seed)
            d2 = create_random_direction(model, args.seed + 1)

        # In a grid from [-x,+x]*[-y,+y] eval as if the weights were shifted to that space
        x_min = -1
        x_max = 1
        num_points = 21
        alphas = np.linspace(x_min, x_max, num_points)
        betas = np.linspace(x_min, x_max, num_points)

        # Grids are indexed [beta, alpha]: rows are beta, columns are alpha
        loss_grid = np.zeros((len(betas), len(alphas)))
        acc_grid = np.zeros((len(betas), len(alphas)))
        for j, beta in enumerate(betas):
            for i, alpha in enumerate(alphas):
                with torch.no_grad():
                    for p, w, u, v in zip(model_shifted.parameters(), theta_star, d1, d2):
                        # Shift the model alpha steps in the direction d1 and beta steps in the direction d2
                        p.copy_(w + float(alpha) * u + float(beta) * v)
                    # Evaluate at that point
                    # TODO[trajectory-plotting] WIth trajectory, plot the loss grid on the same data.
                    loss_grid[j, i], acc_grid[j, i] = evaluate(model_shifted, data.validation, device)
            print(f"2D sweep: finished row {j + 1}/{len(betas)} (beta = {beta:.2f})")

        # Save the raw numbers
        out_path = Path("plots") / f"{checkpoint_path.stem}_2d.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # TODO[trajectory-plotting] Write trajectory into the npz, if present
        np.savez(
            out_path,
            alphas=alphas,
            betas=betas,
            loss=loss_grid,
            acc=acc_grid,
            seed=np.array(args.seed),
            checkpoint=np.array(str(checkpoint_path)),
        )
        print("Saved 2D sweep to", out_path)

if __name__ == "__main__":
    main()
