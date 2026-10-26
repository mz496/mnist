from collections.abc import Iterable

import numpy as np
import torch
from matplotlib import colormaps
from torch.nn import Module

from mnist.filter_image import to_uint8_rgb, tile_row
from mnist.model import LeNet5

def first_example_per_class(batches: Iterable[tuple[torch.Tensor, torch.Tensor]], class_count: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scans (images, labels) batches in order and keeps the first image seen for each class, stopping as soon as every
    class has one. Returns (images [class_count, ...], labels [class_count]) ordered by class, 0 to class_count - 1.
    Pass an unshuffled loader so the same exemplars are picked every time.
    Raises ValueError if some class never appears.
    """
    found: dict[int, torch.Tensor] = {}
    for images, labels in batches:
        for image, label in zip(images, labels.tolist()):
            if label not in found:
                found[label] = image
        if len(found) == class_count:
            break
    missing = [label for label in range(class_count) if label not in found]
    if missing:
        raise ValueError(f"no examples found for classes {missing}")
    return torch.stack([found[label] for label in range(class_count)]), torch.arange(class_count)

def make_capture_hook(name: str, captured: dict[str, torch.Tensor]):
    """
    Returns a forward hook that stores the module's output in `captured[name]`.
    Unlike make_stats_hook / make_stash_hook in train.py, this records in eval mode too, since it is
    only registered for the duration of a probe pass.
    """
    def hook(module: Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured[name] = output.detach()
    return hook

def capture_conv1_maps(model: LeNet5, probe_images: torch.Tensor) -> dict[str, np.ndarray]:
    """
    Runs images [N, 1, 28, 28] through the model in eval mode and returns conv1's output for them, as NumPy:
    - "conv1_preact": conv1's output, the input to relu1
    - "conv1_relu": relu1's output
    Each is [N, C, H, W]. The model's train/eval mode is restored, and the temporary hooks are removed, even on error.

    The permanent training hooks (stats, stash) skip eval mode, so this does not disturb them.
    """
    captured: dict[str, torch.Tensor] = {}
    handles = [
        model.conv1.register_forward_hook(make_capture_hook("conv1_preact", captured)),
        model.relu1.register_forward_hook(make_capture_hook("conv1_relu", captured)),
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
    # One transfer per tensor
    return {name: tensor.cpu().numpy() for name, tensor in captured.items()}

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
