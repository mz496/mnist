
import argparse
import copy
from pathlib import Path
from typing import Any

import torch
import numpy as np

from mnist.config import load_default_config
from mnist.data import get_dataloaders
from mnist.evaluate import evaluate
from mnist.loss_landscape.lib import create_random_direction, get_pca_directions_for_snapshots, project_snapshots_onto_pca_directions
from mnist.model import LeNet5
from mnist.train import load_checkpoint, load_training_snapshots
from mnist.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep a run's best checkpoint's weights along random directions, evaluate the validation loss and accuracy at each point, and save the results to an .npz in plots/"
    )
    parser.add_argument("run_id", type=str, help="run id of a training run, i.e. the name of its folder under checkpoints/, e.g. 2026-10-14T19:28:51_45xyzktb (loads checkpoints/<run_id>/best_validation_loss.pt)")
    parser.add_argument("--trajectory-plot", action="store_true", help="turn on non-random directions mode to plot trajectories (uses the training snapshots' PCA directions for this run id instead, and centers on the final training snapshot rather than the best validation checkpoint)")
    parser.add_argument("--seed", type=int, default=0, help="when in random direction mode, seed for the randomness (2d mode also uses seed + 1)")
    parser.add_argument("--mode", choices=["1d", "2d"], default="1d", help="sweep along one direction or over a 2D grid")
    return parser.parse_args()

def main():
    args = parse_args()
    run_id: str = args.run_id
    device = get_device()
    print("MOVING MODEL TO DEVICE:", device)
    if args.trajectory_plot and args.mode == "1d":
        raise Exception("trajectory plot not supported for 1D")

    if args.trajectory_plot:
        # [n, D] where n is the number of snapshots
        snapshots = load_training_snapshots(run_id)
        # Theta star is the final training snapshot, not the best validation checkpoint
        model = LeNet5().to(device)
        # Copy the final snapshot into the model parameters.
        torch.nn.utils.vector_to_parameters(torch.from_numpy(snapshots[-1]).to(device), model.parameters())
    else:
        # Fetch saved weights from the run's best checkpoint (theta star)
        model = load_checkpoint(run_id, device)
    theta_star: list[torch.Tensor] = [p.detach().clone() for p in model.parameters()]
    # Vector form of theta_star for trajectory plotting, intentionally matching the vectorization of the snapshots
    theta_star_vector: np.ndarray = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()

    model_shifted = copy.deepcopy(model)

    config = load_default_config()
    data = get_dataloaders(config.data)

    if args.mode == "1d":
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
        out_path = Path("plots") / f"{run_id}_1d.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            alphas=alphas,
            loss=np.array(losses),
            acc=np.array(accs),
            seed=np.array(args.seed),
            run_id=np.array(run_id),
        )
        print("Saved 1D sweep to", out_path)

    elif args.mode == "2d":
        # (alpha, beta) of each training snapshot, only set in trajectory mode
        trajectory_points: np.ndarray | None = None

        if args.trajectory_plot:
            # Use training snapshots as the plotting plane and to plot the trajectory on that plane

            # PCA on training snapshots; extract top 2
            snapshot_pca_directions, explained_variance = get_pca_directions_for_snapshots(snapshots)
            print("SNAPSHOT PCA DIRECTIONS")
            print(snapshot_pca_directions)
            print("EXPLAINED VARIANCE")
            print(explained_variance)
            d1 = snapshot_pca_directions[0]
            d2 = snapshot_pca_directions[1]
            
            # Center each snapshot point (snapshots [n, D]) on theta_star [1, D], then project onto PCAs [2, D] to get per-snapshot points [n, 2]
            points = project_snapshots_onto_pca_directions(snapshots, theta_star_vector, snapshot_pca_directions)
            trajectory_points = points
            # float() so the bounds are known to be floats (numpy's min/max results are partly unknown to the type checker)
            x_min, y_min = (float(v) for v in points.min(axis=0))
            x_max, y_max = (float(v) for v in points.max(axis=0))
            # the grid must include the origin (theta_star), and a margin so the path isn't on the border
            x_min, x_max = min(x_min, 0.0), max(x_max, 0.0)
            y_min, y_max = min(y_min, 0.0), max(y_max, 0.0)
            x_pad, y_pad = 0.1 * (x_max - x_min), 0.1 * (y_max - y_min)

            # Grid resolution.
            num_points = 21
            alphas = np.linspace(x_min - x_pad, x_max + x_pad, num_points)
            betas = np.linspace(y_min - y_pad, y_max + y_pad, num_points)
        else:
            # Get two random directions in weight-space of the model
            # These will be our axes for plotting
            print("DIRECTION SEEDS:", args.seed, args.seed + 1)
            d1 = create_random_direction(model, args.seed)
            d2 = create_random_direction(model, args.seed + 1)

            # In a grid from [-x,+x]*[-y,+y] eval as if the weights were shifted to that space
            x_min = -1
            x_max = 1
            # Grid resolution.
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
        out_path = Path("plots") / f"{run_id}_2d.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # The plot reads the "trajectory" key if it's present
        # Typed as Any because numpy's savez stubs would otherwise read a **dict[str, ndarray] as possibly meant for its allow_pickle keyword
        extra: dict[str, Any] = {}
        if trajectory_points is not None:
            extra["trajectory"] = trajectory_points  # [n, 2], (alpha, beta) per snapshot
        np.savez(
            out_path,
            alphas=alphas,
            betas=betas,
            loss=loss_grid,
            acc=acc_grid,
            seed=np.array(args.seed),
            run_id=np.array(run_id),
            **extra,
        )
        print("Saved 2D sweep to", out_path)

if __name__ == "__main__":
    main()
