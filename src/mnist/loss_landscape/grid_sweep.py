
import copy
from pathlib import Path
from typing import Any

import torch
import numpy as np

from mnist.config import load_default_config
from mnist.data import get_dataloaders
from mnist.evaluate import evaluate
from mnist.loss_landscape.lib import create_random_direction, get_pca_directions_for_snapshots, project_snapshots_onto_pca_directions, unflatten_like
from mnist.model import LeNet5
from mnist.train import SNAPSHOT_SPACING, load_checkpoint, load_training_snapshots
from mnist.loss_landscape.grid_sweep_file_manager import grid_sweep_path, save_grid_sweep
from mnist.utils import get_device


def create_or_load_grid_sweep(run_id: str, mode: str, trajectory: bool = False, max_step: int | None = None, seed: int = 0) -> Path:
    """
    The path of the grid sweep for these params, computing and saving it first if it doesn't exist yet.
    The sweep is saved next to the run's checkpoint (see grid_sweep_path). Delete the file to compute it again.
    """
    # Round max_step down first, as create_grid_sweep does, so the path looked for is the path it saves to
    if max_step is not None:
        max_step = max_step // SNAPSHOT_SPACING * SNAPSHOT_SPACING
    path = grid_sweep_path(run_id, mode, trajectory=trajectory, max_step=max_step, seed=seed)
    if path.is_file():
        return path
    return create_grid_sweep(run_id, mode, trajectory, max_step, seed)


def create_grid_sweep(run_id: str, mode: str, trajectory: bool = False, max_step: int | None = None, seed: int = 0) -> Path:
    """
    Sweep a run's weights over a line (mode "1d") or a grid (mode "2d"), evaluate the loss and accuracy at each point,
    and save the results. With trajectory, the 2d grid is laid out along the PCA directions of the run's training
    snapshots instead of random directions, and centered on the final snapshot rather than the best validation checkpoint.
    max_step limits trajectory to the snapshots up to that step, rounded down to a multiple of the snapshot spacing.
    The seed picks the random directions, and does nothing for a trajectory.
    Returns the path it saved to.
    """
    if mode not in ("1d", "2d"):
        raise ValueError(f"mode must be '1d' or '2d', got {mode!r}")
    if trajectory and mode == "1d":
        raise ValueError("trajectory plot not supported for 1D")
    if max_step is not None and not trajectory:
        raise ValueError("max_step only applies with trajectory")
    device = get_device()
    print("MOVING MODEL TO DEVICE:", device)

    # The last training step of the window of snapshots used, only set with --max-step. It can be below --max-step if
    # that wasn't a multiple of the snapshot spacing.
    window_last_step: int | None = None

    if trajectory:
        # [n, D] where n is the number of snapshots
        snapshots = load_training_snapshots(run_id)
        if max_step is not None:
            # Snapshot row i is training step (i + 1) * SNAPSHOT_SPACING, so the steps up to max_step are the first rows
            n_keep = max_step // SNAPSHOT_SPACING
            if not 3 <= n_keep <= len(snapshots):
                raise ValueError(f"--max-step {max_step} keeps {n_keep} snapshots, but it needs between 3 and {len(snapshots)}")
            snapshots = snapshots[:n_keep]
            window_last_step = n_keep * SNAPSHOT_SPACING
        # Theta star is the final training snapshot (of the window, if --max-step is set), not the best validation checkpoint
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

    if mode == "1d":
        # Get a random direction in weight-space of the model
        # This will be our axis for plotting
        print("DIRECTION SEED:", seed)
        d1 = create_random_direction(model, seed)

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
        out_path = grid_sweep_path(run_id, mode, seed=seed)
        save_grid_sweep(
            out_path,
            alphas=alphas,
            loss=np.array(losses),
            acc=np.array(accs),
            seed=np.array(seed),
            run_id=np.array(run_id),
            split=np.array("validation"),  # which data the loss and accuracy were measured on
            mode=np.array(mode),  # "1d": the plot reads this to know what kind of sweep the file is
        )
        print("Saved 1D sweep to", out_path)
        return out_path

    elif mode == "2d":
        # Extra arrays for the npz that only exist in trajectory mode: the training path and the PCA explained variance.
        # The plot reads these keys if they're present.
        # Typed as Any because numpy's savez stubs would otherwise read a **dict[str, ndarray] as possibly meant for its allow_pickle keyword
        extra: dict[str, Any] = {}

        # In trajectory mode, measure the loss on a fixed training subset: it's the loss the optimizer was descending.
        # Otherwise use the validation set.
        split = "train" if trajectory else "validation"
        eval_loader = data.train_subset if trajectory else data.validation

        if trajectory:
            # Use training snapshots as the plotting plane and to plot the trajectory on that plane

            # PCA on training snapshots; extract top 2
            snapshot_pca_directions, explained_variance = get_pca_directions_for_snapshots(snapshots)
            print("SNAPSHOT PCA DIRECTIONS")
            print(snapshot_pca_directions)
            print("EXPLAINED VARIANCE")
            print(explained_variance)
            # The components are flat [D] vectors. The grid loop zips over per-layer tensors, as create_random_direction gives,
            # so split each one into one tensor per model parameter.
            d1 = unflatten_like(snapshot_pca_directions[0], model)
            d2 = unflatten_like(snapshot_pca_directions[1], model)
            
            # Center each snapshot point (snapshots [n, D]) on theta_star [1, D], then project onto PCAs [2, D] to get per-snapshot points [n, 2]
            points = project_snapshots_onto_pca_directions(snapshots, theta_star_vector, snapshot_pca_directions)
            extra["trajectory"] = points  # [n, 2], (alpha, beta) per snapshot
            extra["explained_variance"] = explained_variance  # [2], fraction of the snapshots' variance along PC1, PC2
            if window_last_step is not None:
                extra["max_step"] = np.array(window_last_step)
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
            print("DIRECTION SEEDS:", seed, seed + 1)
            d1 = create_random_direction(model, seed)
            d2 = create_random_direction(model, seed + 1)

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
                    loss_grid[j, i], acc_grid[j, i] = evaluate(model_shifted, eval_loader, device)
            print(f"2D sweep: finished row {j + 1}/{len(betas)} (beta = {beta:.2f})")

        # Save the raw numbers
        out_path = grid_sweep_path(run_id, mode, trajectory=trajectory, max_step=window_last_step, seed=seed)
        save_grid_sweep(
            out_path,
            alphas=alphas,
            betas=betas,
            loss=loss_grid,
            acc=acc_grid,
            seed=np.array(seed),
            run_id=np.array(run_id),
            split=np.array(split),  # which data the loss and accuracy were measured on
            mode=np.array(mode),  # "2d": the plot reads this to know what kind of sweep the file is
            **extra,
        )
        print("Saved 2D sweep to", out_path)
        return out_path
    raise AssertionError(mode)
