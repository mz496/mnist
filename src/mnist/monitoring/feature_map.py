from collections.abc import Iterable, Sequence
from typing import NamedTuple

import numpy as np
import torch
import wandb
from matplotlib import colormaps
from torch.nn import Module

from mnist.monitoring.image_utils import tile_row, to_uint8_rgb

# Keys are f"{NAMESPACE}_{layer}/{view name}", so each layer gets its own W&B section.
NAMESPACE = "maps"
METRIC_GLOB = f"{NAMESPACE}_*"
# The prefix of this monitor's tensors in the dict that goes through to_cpu_numpy.
# After it come the layer and the view name, each after a slash.
DEVICE_KEY = "feature_maps"

class FeatureMapView(NamedTuple):
    """
    Description of a module to output as a feature map.
    
    - layer: the section name in W&B, e.g. "conv1".
    - name: the image name inside the section, e.g. "preact".
    - module: the module whose output is read. Its output must be [N, C, H, W].
    - diverging: True for signed values such as pre-ReLU outputs, False for non-negative ones such as post-ReLU outputs.
    """
    layer: str
    name: str
    module: Module
    diverging: bool

def first_example_per_class(batches: Iterable[tuple[torch.Tensor, torch.Tensor]], class_count: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scans (images, labels) batches in order and keeps the first image seen for each class, stopping as soon as every
    class has one.
    
    Returns (images [class_count, ...], labels [class_count]) ordered by class, 0 to class_count - 1.

    - Pass an unshuffled loader so the same exemplars are picked every time.
    - Raises ValueError if some class never appears.
    """
    found: dict[int, torch.Tensor] = {}
    for images, labels in batches:
        for image, label in zip(images, labels):
            if int(label) not in found:
                found[int(label)] = image
        if len(found) == class_count:
            break
    missing = [label for label in range(class_count) if label not in found]
    if missing:
        raise ValueError(f"no examples found for classes {missing}")
    return torch.stack([found[label] for label in range(class_count)]), torch.arange(class_count)

def make_capture_hook(name: str, captured: dict[str, torch.Tensor]):
    """
    Returns a forward hook that stores the module's output in `captured[name]`.
    Unlike the stats and histogram hooks, this records in eval mode too, since it is
    only registered for the duration of a probe pass.
    """
    def hook(module: Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured[name] = output.detach()
    return hook

def collect_feature_maps(model: Module, probe_images: torch.Tensor, views: Sequence[FeatureMapView]) -> dict[str, torch.Tensor]:
    """
    Runs `probe_images` [N, 1, 28, 28] through the model in eval mode and returns each view's output, [N, C, H, W],
    as detached device tensors keyed for to_cpu_numpy. The model's train/eval mode is restored, and the temporary hooks
    are removed, even on error.

    The permanent training hooks skip eval mode, so this does not disturb them.
    """
    captured: dict[str, torch.Tensor] = {}
    handles = [
        view.module.register_forward_hook(make_capture_hook(f"{DEVICE_KEY}/{view.layer}/{view.name}", captured))
        for view in views
    ]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(probe_images)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return captured

def feature_maps_to_image(original: np.ndarray, maps: np.ndarray, diverging: bool, scale: int = 4, gap: int = 2) -> np.ndarray:
    """
    One row: the original image, then each channel's map. original [H, W], maps [C, H, W] -> uint8 RGB.
    Each channel is normalized on its own, so a high-scale channel does not wash out the others.

    diverging: for signed values (pre-ReLU). Red positive, blue negative, white zero, scaled by the channel's largest absolute value.
               Otherwise (post-ReLU) a sequential colormap from each channel's min to its max, so a flat channel is one solid color.
    """
    if diverging:
        limit = np.maximum(np.abs(maps).max(axis=(1, 2), keepdims=True), 1e-12)  # [C, 1, 1]
        unit = 0.5 + 0.5 * np.clip(maps / limit, -1.0, 1.0)
        rgb_maps = to_uint8_rgb(colormaps["RdBu_r"](unit))
    else:
        low = maps.min(axis=(1, 2), keepdims=True)
        high = maps.max(axis=(1, 2), keepdims=True)
        unit = (maps - low) / np.maximum(high - low, 1e-12)
        rgb_maps = to_uint8_rgb(colormaps["viridis"](unit))

    original_range = max(float(original.max() - original.min()), 1e-12)
    rgb_original = to_uint8_rgb(colormaps["gray"]((original - original.min()) / original_range))[None]  # [1, H, W, 3]
    return tile_row(np.concatenate([rgb_original, rgb_maps]), scale, gap)

def feature_maps_grid(originals: np.ndarray, maps: np.ndarray, diverging: bool, scale: int = 4, gap: int = 2) -> np.ndarray:
    """
    One row per image, stacked top to bottom with `gap` white pixels between rows (see feature_maps_to_image).
    originals [N, H, W], maps [N, C, H, W] -> uint8 RGB.
    Normalization is per image and per channel, so absolute scales are not comparable between rows.
    """
    rows = [feature_maps_to_image(original, image_maps, diverging, scale, gap) for original, image_maps in zip(originals, maps)]
    spacer = np.full((gap, rows[0].shape[1], 3), 255, dtype=np.uint8)
    stacked: list[np.ndarray] = []
    for index, row in enumerate(rows):
        if index > 0:
            stacked.append(spacer)
        stacked.append(row)
    return np.concatenate(stacked, axis=0)

def log_feature_maps(
    host: dict[str, np.ndarray],
    views: Sequence[FeatureMapView],
    originals: np.ndarray,
    digits: str,
    step_number: int,
) -> dict[str, wandb.Image]:
    """
    One image per view: a row per probe image, top to bottom, in the order of `originals`.
    host: the result of to_cpu_numpy, entries of other monitors are ignored.
    originals: [N, H, W] probe images for the left column. digits: their labels, for the caption.
    """
    caption = f"step {step_number}; digits {digits}"
    return {
        f"{NAMESPACE}_{view.layer}/{view.name}": wandb.Image(
            feature_maps_grid(originals, host[f"{DEVICE_KEY}/{view.layer}/{view.name}"], view.diverging), caption=caption
        )
        for view in views
    }

def define_feature_map_metrics() -> None:
    wandb.define_metric(METRIC_GLOB, step_metric="step_number")
