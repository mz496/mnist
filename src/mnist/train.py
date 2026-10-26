from pathlib import Path
from typing import cast

import numpy as np
import wandb

from mnist.config import Config, load_default_config
from mnist.data import get_dataloaders
from mnist.evaluate import evaluate
from mnist.monitoring.activation_histogram import (
    collect_activation_histograms,
    define_activation_histogram_metrics,
    log_activation_histograms,
    register_activation_histograms,
)
from mnist.monitoring.activation_summary_stats import (
    collect_activation_summary_stats,
    define_activation_summary_stats_metrics,
    log_activation_summary_stats,
    register_activation_summary_stats,
)
from mnist.monitoring.feature_map import (
    FeatureMapView,
    collect_feature_maps,
    define_feature_map_metrics,
    first_example_per_class,
    log_feature_maps,
)
from mnist.monitoring.filter_image import collect_filter_weights, define_filter_image_metrics, log_filter_images
from mnist.monitoring.layer_view import LayerViewSpec, define_layer_view_metrics, log_layer_views
from mnist.monitoring.transfer import to_cpu_numpy
from mnist.type_aliases import CrossEntropyLossFunction
from mnist.utils import set_seed, get_device, timestamp
from mnist.model import LeNet5
import torch
from torch.nn import Module

# Steps between training snapshots. Snapshot row i is training step (i + 1) * SNAPSHOT_SPACING,
# which is how loss_landscape/plot/grid_sweep turns a step number into a row count.
SNAPSHOT_SPACING = 10

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
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"PARAMETER COUNT CHECK: {parameter_count}")

    # *_layers dicts describe what we are registering for each monitor type
    stats_layers: dict[str, Module] = {
        "conv1": model.conv1,
        "relu1": model.relu1,
        "conv2": model.conv2,
        "relu2": model.relu2,
        "fc1": model.fc1,
        "relu3": model.relu3,
        "fc2": model.fc2,
        "relu4": model.relu4,
        "fc3": model.fc3,
    }
    histogram_layers: dict[str, Module] = {"conv1": model.conv1, "conv2": model.conv2}
    filter_layers = {"conv1": model.conv1, "conv2": model.conv2}
    # The fixed color limit for the filter images. The default init shrinks with the number of inputs
    # (conv1 starts within +-0.2, conv2 within +-0.08), so conv2 gets a smaller limit. The conv2 value is a first guess.
    filter_vmax_by_layer = {"conv1": 1.0, "conv2": 0.5}
    # The left of each row is the input to the layer's convolution: the digit for conv1, and conv1's pooled output
    # for conv2. Both sides of a row use the same scale, so conv2's 14 by 14 input is taller than its 10 by 10 maps
    # and the maps are centered on white.
    feature_map_views = [
        FeatureMapView("conv1", "preact", model.conv1, input_from=model.conv1, diverging=True),
        FeatureMapView("conv1", "relu", model.relu1, input_from=model.conv1, diverging=False),
        FeatureMapView("conv2", "preact", model.conv2, input_from=model.conv2, diverging=True, scale=5),
        FeatureMapView("conv2", "relu", model.relu2, input_from=model.conv2, diverging=False, scale=5),
    ]
    # The combined images: each layer's weights drawn above one of its feature map views, with the output channels lined up.
    # Each layer must be in both the filter layers and the feature map views above.
    layer_view_specs = [LayerViewSpec("conv1", "preact"), LayerViewSpec("conv2", "preact")]

    # Activation monitoring: each monitor's hooks overwrite their entries every forward pass.
    activation_stats, stats_handles = register_activation_summary_stats(stats_layers)
    stashed_outputs, histogram_handles = register_activation_histograms(histogram_layers)
    hook_handles = stats_handles + histogram_handles

    # The filters at init, to log how far training has moved each one. A one-time copy to the host.
    initial_filter_weights = to_cpu_numpy(collect_filter_weights(filter_layers))

    optimizer = torch.optim.SGD(model.parameters(), lr=config.train.learning_rate, momentum=config.train.momentum)
    loss_fn = cast(CrossEntropyLossFunction, torch.nn.CrossEntropyLoss())

    # W&B plumbing
    name = timestamp()
    if existing_run is None:
        run = wandb.init(project="mnist", name=tag or name, group=group, config=config.model_dump())
    else:
        run = existing_run
    # A unique and sortable run ID to identify this run.
    run_id = f"{name}_{run.id}"
    wandb.define_metric("step_number", hidden=True)
    wandb.define_metric("train/loss", step_metric="step_number")
    wandb.define_metric("validation/loss", step_metric="step_number")
    wandb.define_metric("validation/accuracy", step_metric="step_number")
    define_activation_summary_stats_metrics()
    define_activation_histogram_metrics()
    define_filter_image_metrics()
    define_feature_map_metrics()
    define_layer_view_metrics()

    data = get_dataloaders(config.data)

    # Fixed probe images: the first validation example of each digit 0-9 (the validation loader is unshuffled),
    # so the feature maps only change because the model does. Rows are in digit order.
    probe_cpu_images, probe_labels = first_example_per_class(data.validation)  # [10, 1, 28, 28], [10]
    probe_images = probe_cpu_images.to(device)
    probe_label_names = [str(int(label)) for label in probe_labels]

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

                # Gather everything that lives on the device, copy it to the host in one place, then log from NumPy.
                # The weights are read after optimizer.step(), so they are one update newer than the activations.
                stat_names, stats_device = collect_activation_summary_stats(activation_stats)
                device_tensors = {
                    **stats_device,
                    **collect_activation_histograms(stashed_outputs),
                    **collect_filter_weights(filter_layers),
                    **collect_feature_maps(model, probe_images, feature_map_views),
                }
                host = to_cpu_numpy(device_tensors)

                wandb.log({
                    "train/loss": train_loss_value,
                    **log_activation_summary_stats(stat_names, host),
                    **log_activation_histograms(host),
                    **log_filter_images(host, initial_filter_weights, step_number, filter_vmax_by_layer),
                    **log_feature_maps(host, feature_map_views, probe_label_names, step_number),
                    **log_layer_views(host, layer_view_specs, feature_map_views, probe_label_names, step_number, filter_vmax_by_layer),
                    "step_number": step_number,
                })

            if save_snapshots and step_number % SNAPSHOT_SPACING == 0:
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
