from pathlib import Path
from typing import cast

import numpy as np
import wandb

from mnist.config import load_default_config
from mnist.data import get_dataloaders
from mnist.evaluate import evaluate
from mnist.type_aliases import CrossEntropyLossFunction
from mnist.utils import set_seed, get_device
from mnist.model import LeNet5
import torch
from torch.nn import Module
from datetime import datetime

# Steps between training snapshots. Snapshot row i is training step (i + 1) * SNAPSHOT_SPACING,
# which is how loss_landscape/plot/grid_sweep turns a step number into a row count.
SNAPSHOT_SPACING = 10

def main():
    config = load_default_config()
    print("USING MODEL CONFIG")
    print(config)

    set_seed(config.train.seed)

    device = get_device()
    print(f"MOVING MODEL TO DEVICE: {device}")
    model = LeNet5().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"PARAMETER COUNT CHECK: {parameter_count}")

    optimizer = torch.optim.SGD(model.parameters(), lr=config.train.learning_rate, momentum=config.train.momentum)
    loss_fn = cast(CrossEntropyLossFunction, torch.nn.CrossEntropyLoss())

    # W&B plumbing
    name = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    run = wandb.init(project="mnist", name=name, config=config.model_dump())
    # A unique and sortable run ID to identify this run.
    run_id = f"{name}_{run.id}"
    wandb.define_metric("step_number", hidden=True)
    wandb.define_metric("train/loss", step_metric="step_number")
    wandb.define_metric("validation/loss", step_metric="step_number")
    wandb.define_metric("validation/accuracy", step_metric="step_number")

    data = get_dataloaders(config.data)

    epoch_count = config.train.epoch_count
    step_number = 0
    best_validation_loss = float("inf")

    # For trajectory mapping:
    # Fill a snapshot buffer, fill it, then write it to disk at the end
    # We could preallocate but the calculation is too annoying.

    # Each row is a weight snapshot, taken every SNAPSHOT_SPACING steps.
    training_snapshot_vectors: list[torch.Tensor] = []

    for epoch in range(epoch_count):
        for batch in data.train:
            # Set train mode on the model
            model.train()

            # We could also have copied this all over upfront for MNIST, since it's small
            batch_images, batch_labels = batch
            images: torch.Tensor = batch_images.to(device)
            labels: torch.Tensor = batch_labels.to(device)

            # Compute loss
            logits = model(images)
            train_loss = loss_fn(logits, labels)

            # Zero the gradients for this step
            optimizer.zero_grad()

            # Backprop the gradients through the computation graph
            # TODO 10-13: Fix type stubs.
            train_loss.backward()

            # Update weights using the computed gradients
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]

            # Log training loss every Nth batch/step
            step_number += 1
            if step_number % 10 == 0:
                # A sync point explicitly moving data from the GPU to the CPU
                train_loss_value = train_loss.item()
                wandb.log({"train/loss": train_loss_value, "step_number": step_number})
            
            if step_number % SNAPSHOT_SPACING == 0:
                # parameters_to_vector takes a copy, then we detach it from the autograd graph, then move it to the CPU
                training_snapshot_vectors.append(torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu())
        
        # Log validation loss once per epoch
        validation_loss, validation_accuracy = evaluate(model, data.validation, device)
        wandb.log({"validation/loss": validation_loss, "validation/accuracy": validation_accuracy, "step_number": step_number})

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            print(f"updated best checkpoint after achieving validation loss {best_validation_loss}")
            save_checkpoint(model, run_id)

        # Write out snapshots
        save_training_snapshots(run_id, training_snapshot_vectors)

        print(f"completed epoch {epoch}")

def save_training_snapshots(run_id: str, training_snapshot_vectors: list[torch.Tensor]) -> None:
    """
    Writes out training snapshots to an npz for the trajectory mapping to read
    """
    run_dir = Path("checkpoints") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        run_dir / "training_snapshot_vectors.npz",
        snapshot_vectors=torch.stack(training_snapshot_vectors).numpy(),
    )

def load_training_snapshots(run_id: str) -> np.ndarray:
    """
    Reads the snapshots written by save_training_snapshots.
    Returns [n_snapshots, parameter_count] float32: one row per snapshot, in training order.
    """
    path = Path("checkpoints") / run_id / "training_snapshot_vectors.npz"
    if not path.is_file():
        raise FileNotFoundError(f"training snapshots not found: {path}")
    with np.load(path) as data:
        snapshots = data["snapshot_vectors"]
    if snapshots.ndim != 2:
        raise ValueError(f"expected a [n_snapshots, parameter_count] matrix, got shape {snapshots.shape}")
    return snapshots

def best_checkpoint_path(run_id: str) -> Path:
    return Path("checkpoints") / run_id / "best_validation_loss.pt"

def save_checkpoint(model: Module, run_id: str) -> None:
    """
    torch.save s the best-validation-loss checkpoint for run_id.
    """
    path = best_checkpoint_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(run_id: str, device: str) -> LeNet5:
    """
    Reads the best-validation-loss checkpoint written by save_checkpoint into a fresh model on `device`.
    """
    path = best_checkpoint_path(run_id)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    model = LeNet5().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

if __name__ == "__main__":
    main()
