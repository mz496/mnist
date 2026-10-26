from pathlib import Path
from typing import cast

import numpy as np
import wandb

from mnist.config import Config, load_default_config
from mnist.data import get_dataloaders
from mnist.evaluate import evaluate
from mnist.filter_image import FILTER_VMAX, filters_to_image
from mnist.type_aliases import CrossEntropyLossFunction
from mnist.utils import set_seed, get_device, timestamp
from mnist.model import LeNet5
import torch
from torch.nn import Module

# Steps between training snapshots. Snapshot row i is training step (i + 1) * SNAPSHOT_SPACING,
# which is how loss_landscape/plot/grid_sweep turns a step number into a row count.
SNAPSHOT_SPACING = 10

# Bins per W&B histogram (default 64, max 512). Fine bins help resolve the thin background spike in conv1.
HISTOGRAM_BINS = 256

def channel_histograms(layer_name: str, output: np.ndarray) -> dict[str, wandb.Histogram]:
    """
    One W&B histogram per channel of a [N, C, H, W] array, over every image and position.
    The bin range is each channel's own min/max at that step, so bin edges differ between steps.
    """
    return {
        f"activations_{layer_name}_preact_hist/ch{channel}": wandb.Histogram(
            np_histogram=np.histogram(output[:, channel], bins=HISTOGRAM_BINS)
        )
        for channel in range(output.shape[1])
    }

def make_stats_hook(name: str, stats: dict[str, torch.Tensor]):
    """
    Returns a forward hook that records status into `stats` as a dict of key -> 0D tensor.
    - Overwrites on every step.
    - Does not move any values across devices.
    - Only records in train mode.
    """
    @torch.no_grad()
    def hook(module: Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not module.training:
            return
        stats[f"activations_{name}/mean"] = output.mean()
        stats[f"activations_{name}/std"] = output.std()
        if isinstance(module, torch.nn.ReLU):
            stats[f"activations_{name}/frac_zero_mean"] = (output == 0).float().mean()
            # Only look at relus downstream of the convolutions: N, C, H, W.
            if output.dim() == 4:
                zeros = (output == 0).float()
                mean_by_channel = zeros.mean(dim=[0, 2, 3]) # [C]
                for channel in range(mean_by_channel.numel()):
                    stats[f"activations_{name}/frac_zero_ch{channel}"] = mean_by_channel[channel]
        if isinstance(module, torch.nn.Tanh):
            stats[f"activations_{name}/frac_saturated"] = (output.abs() > 0.95).float().mean()
            stats[f"activations_{name}/tanh_grad_mean"] = (1 - output**2).float().mean()
    return hook

def make_stash_hook(name: str, stash: dict[str, torch.Tensor]):
    """
    Returns a forward hook that stashes the module's entire raw output in `tensors[name]`.
    - Overwrites on every step.
    - Used to build histograms at logging time. Stores a reference only: no copy, no device transfer.
    - Only records in train mode.
    """
    def hook(module: Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not module.training:
            return
        # Detach from the autograd graph when adding to the stash.
        stash[name] = output.detach()
    return hook

def train(
    config: Config,
    group: str | None = None, 
    tag: str | None = None, 
    existing_run: wandb.Run | None = None, 
    save_snapshots: bool = True,
    ) -> str:
    """
    Trains one model with `config` and returns its run_id.

    - `group` groups runs together in W&B
    - `tag` is the W&B display name (defaults to the timestamp).
    - `existing_run` if we want to reuse an existing run that the caller already initialized (e.g. W&B's sweep feature)
    - `save_snapshots` controls whether weight snapshots for the loss landscape plots are collected and written.
    """
    print("USING MODEL CONFIG")
    print(config)

    set_seed(config.train.seed)

    device = get_device()
    print(f"MOVING MODEL TO DEVICE: {device}")
    model = LeNet5().to(device)
    # conv1's weights at init, to log how far training has moved each filter. Copied so later updates don't change it.
    conv1_initial_weights = model.conv1.weight.detach().cpu().numpy().copy()
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"PARAMETER COUNT CHECK: {parameter_count}")

    # Activation monitoring: each hook overwrites its entries every forward pass.
    activation_stats: dict[str, torch.Tensor] = {}
    # Raw layer outputs kept around for per-channel histograms, [N, C, H, W].
    stashed_outputs: dict[str, torch.Tensor] = {}
    hook_handles = [
        model.conv1.register_forward_hook(make_stash_hook("conv1", stashed_outputs)),
        model.conv1.register_forward_hook(make_stats_hook("conv1", activation_stats)),
        model.relu1.register_forward_hook(make_stats_hook("relu1", activation_stats)),
        model.conv2.register_forward_hook(make_stash_hook("conv2", stashed_outputs)),
        model.conv2.register_forward_hook(make_stats_hook("conv2", activation_stats)),
        model.relu2.register_forward_hook(make_stats_hook("relu2", activation_stats)),
        model.fc1.register_forward_hook(make_stats_hook("fc1", activation_stats)),
        model.relu3.register_forward_hook(make_stats_hook("relu3", activation_stats)),
        model.fc2.register_forward_hook(make_stats_hook("fc2", activation_stats)),
        model.relu4.register_forward_hook(make_stats_hook("relu4", activation_stats)),
        model.fc3.register_forward_hook(make_stats_hook("fc3", activation_stats)),
    ]

    optimizer = torch.optim.SGD(model.parameters(), lr=config.train.learning_rate, momentum=config.train.momentum)
    loss_fn = cast(CrossEntropyLossFunction, torch.nn.CrossEntropyLoss())

    # W&B plumbing
    name = timestamp()
    if existing_run is None:
        run = wandb.init(project="mnist", name=tag or name, group=group, config=config.model_dump())
    # A unique and sortable run ID to identify this run.
    run_id = f"{name}_{run.id}"
    wandb.define_metric("step_number", hidden=True)
    wandb.define_metric("train/loss", step_metric="step_number")
    wandb.define_metric("validation/loss", step_metric="step_number")
    wandb.define_metric("validation/accuracy", step_metric="step_number")
    wandb.define_metric("activations_*", step_metric="step_number")
    wandb.define_metric("weights_*", step_metric="step_number")

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
                print(f"step {step_number:03d}: train loss {train_loss_value:.3f}")

                # Activation stats per layer
                # Each is a 0D tensor; structure as one tensor, transfer it from CPU -> CPU once, then destructure it again
                stat_names = list(activation_stats)
                stat_values_tensor = torch.stack(list(activation_stats.values()))
                # One GPU->CPU sync for all the activation stats together
                stat_cpu = stat_values_tensor.cpu()
                stat_values: list[torch.Tensor] = stat_cpu.tolist()

                # Per-channel histograms of the convs' preactivations
                # Move each whole [N, C, H, W] tensor to the CPU once, then slice channels in NumPy.
                conv1_histograms = channel_histograms("conv1", stashed_outputs["conv1"].cpu().numpy())
                conv2_histograms = channel_histograms("conv2", stashed_outputs["conv2"].cpu().numpy())

                # conv1's six 5x5 filters as images:
                # - fixed scale: magnitudes comparable across steps
                # - normalized: each filter's pattern visible from the start
                # - change from init (normalized): only what training has added, without the random starting pattern
                # Read after optimizer.step(), so these are one update newer than the forward pass the activation stats came from.
                conv1_weights = model.conv1.weight.detach().cpu().numpy()
                filter_caption = f"step {step_number}"
                conv1_filter_images = {
                    "weights_conv1/filters_fixed_scale": wandb.Image(filters_to_image(conv1_weights, FILTER_VMAX), caption=filter_caption),
                    "weights_conv1/filters_normalized": wandb.Image(filters_to_image(conv1_weights, None), caption=filter_caption),
                    "weights_conv1/filters_change_from_init": wandb.Image(
                        filters_to_image(conv1_weights - conv1_initial_weights, None), caption=filter_caption
                    ),
                }

                wandb.log({
                    "train/loss": train_loss_value,
                    **dict(zip(stat_names, stat_values)),
                    **conv1_histograms,
                    **conv2_histograms,
                    **conv1_filter_images,
                    "step_number": step_number,
                })

            if save_snapshots and step_number % SNAPSHOT_SPACING == 0:
                # parameters_to_vector takes a copy, then we detach it from the autograd graph, then move it to the CPU
                training_snapshot_vectors.append(torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu())

            if step_number == 5000:
                raise Exception("Stopping")
        
        # Log validation loss once per epoch
        validation_loss, validation_accuracy = evaluate(model, data.validation, device)
        wandb.log({"validation/loss": validation_loss, "validation/accuracy": validation_accuracy, "step_number": step_number})

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            print(f"updated best checkpoint after achieving validation loss {best_validation_loss}")
            save_checkpoint(model, run_id)

        # Write out snapshots
        if save_snapshots:
            save_training_snapshots(run_id, training_snapshot_vectors)

        print(f"completed epoch {epoch}")

    # Cleanup hygiene for hooks
    for handle in hook_handles:
        handle.remove()

    # Only finish the run if we created it ourselves.
    if existing_run is None:
        wandb.finish()
    return run_id

def main():
    train(load_default_config())

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
