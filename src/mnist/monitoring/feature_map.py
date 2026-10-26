from collections.abc import Iterable, Sequence
from typing import NamedTuple

import numpy as np
import torch
import wandb
from matplotlib import colormaps
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from mnist.monitoring.image_utils import ColumnGroup, add_axis_labels, join_row, stack_rows, tile_row, to_uint8_rgb

# Keys are f"{NAMESPACE}_{layer}/{view name}", so each layer gets its own W&B section.
NAMESPACE = "maps"
METRIC_GLOB = f"{NAMESPACE}_*"
# The prefix of this monitor's tensors in the dict that goes through to_cpu_numpy.
# After it come the layer and either a view name or the word input, each after a slash.
DEVICE_KEY = "feature_maps"

class FeatureMapView(NamedTuple):
    """
    Description of one image to log: a module's output, drawn next to a module's input.

    - layer: the W&B section the image goes in. Views with the same layer share a section.
    - name: the image's name inside the section.
    - module: the module whose output is drawn. Its output must be [N, C, H, W].
    - input_from: the module whose input is drawn on the left of each row. Its input must be [N, C_in, H_in, W_in].
      The input is captured once per layer, so views with the same layer should use the same module here.
    - diverging: True if the output can be negative, which is drawn red and blue around a white zero.
      False if it is never negative, which is drawn on a sequential colormap from its minimum to its maximum.
    - scale: each value is drawn as a scale x scale block of pixels, upscaled with nearest-neighbor.
      Both input and output are shown at the same scale.
    """
    layer: str
    name: str
    module: Module
    input_from: Module
    diverging: bool
    scale: int = 4

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

def make_output_capture_hook(name: str, captured: dict[str, torch.Tensor]):
    """
    Returns a forward hook that stores the module's output in `captured[name]`.
    Unlike the stats and histogram hooks, this records in eval mode too, since it is
    only registered for the duration of a probe pass.
    """
    def hook(module: Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured[name] = output.detach()
    return hook

def make_input_capture_hook(name: str, captured: dict[str, torch.Tensor]):
    """Like make_output_capture_hook, but stores the module's first input."""
    def hook(module: Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured[name] = inputs[0].detach()
    return hook

def collect_feature_maps(model: Module, probe_images: torch.Tensor, views: Sequence[FeatureMapView]) -> dict[str, torch.Tensor]:
    """
    Runs `probe_images` [N, ...] through the model in eval mode and returns, as detached device tensors keyed
    for to_cpu_numpy:
    - each view's output, [N, C, H, W]
    - each layer's input, [N, C_in, H, W], read from the first view of that layer's `input_from` module
    The model's train/eval mode is restored, and the temporary hooks are removed, even on error.

    The permanent training hooks skip eval mode, so this does not disturb them.
    """
    captured: dict[str, torch.Tensor] = {}
    handles: list[RemovableHandle] = []
    layers_with_input: set[str] = set()
    for view in views:
        handles.append(view.module.register_forward_hook(make_output_capture_hook(f"{DEVICE_KEY}/{view.layer}/{view.name}", captured)))
        if view.layer not in layers_with_input:
            layers_with_input.add(view.layer)
            handles.append(view.input_from.register_forward_hook(make_input_capture_hook(f"{DEVICE_KEY}/{view.layer}/input", captured)))
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

def feature_maps_to_image(
    inputs: np.ndarray,
    maps: np.ndarray,
    diverging: bool,
    scale: int = 4,
    gap: int = 2,
    section_gap: int = 8,
) -> np.ndarray:
    """
    One row for one image: the layer's input tiles, a wider gap, then the output map of each channel.
    inputs [C_in, H_in, W_in], maps [C, H, W] -> uint8 RGB.
    Both sides use the same scale. Every tile gets a cell as wide as the wider of the input tile and the map tile,
    and a narrower tile is centered in its cell on white. So all the columns are the same width. If the tiles also
    differ in height, the shorter side is centered vertically on white.
    Each channel is normalized on its own, so a high-scale channel does not wash out the others.

    The inputs are drawn in grayscale from each channel's min to its max.
    scale: each input value and each map value is drawn as a scale x scale block of pixels.
    gap: white pixels between tiles on one side. section_gap: white pixels between the input side and the maps side.
    diverging: if True the maps are drawn red for positive, blue for negative and white for zero, scaled by the channel's
               largest absolute value. If False they are drawn on a sequential colormap from each channel's min to its max,
               so a flat channel is one solid color.
    """
    input_low = inputs.min(axis=(1, 2), keepdims=True)
    input_high = inputs.max(axis=(1, 2), keepdims=True)
    input_unit = np.asarray((inputs - input_low) / np.maximum(input_high - input_low, 1e-12), dtype=np.float32)
    cell_width = max(inputs.shape[2], maps.shape[2]) * scale
    inputs_part = tile_row(to_uint8_rgb(colormaps["gray"](input_unit)), scale, gap, cell_width)

    if diverging:
        limit = np.maximum(np.abs(maps).max(axis=(1, 2), keepdims=True), 1e-12)  # [C, 1, 1]
        unit = 0.5 + 0.5 * np.clip(maps / limit, -1.0, 1.0)
        rgb_maps = to_uint8_rgb(colormaps["RdBu_r"](unit))
    else:
        low = maps.min(axis=(1, 2), keepdims=True)
        high = maps.max(axis=(1, 2), keepdims=True)
        unit = (maps - low) / np.maximum(high - low, 1e-12)
        rgb_maps = to_uint8_rgb(colormaps["viridis"](unit))
    maps_part = tile_row(rgb_maps, scale, gap, cell_width)
    return join_row([inputs_part, maps_part], section_gap)

def feature_maps_grid(
    inputs: np.ndarray,
    maps: np.ndarray,
    diverging: bool,
    scale: int = 4,
    gap: int = 2,
) -> np.ndarray:
    """
    One row per image, stacked top to bottom with `gap` white pixels between rows (see feature_maps_to_image).
    inputs [N, C_in, H_in, W_in], maps [N, C, H, W] -> uint8 RGB.
    Normalization is per image and per channel, so absolute scales are not comparable between rows.
    """
    rows = [feature_maps_to_image(image_inputs, image_maps, diverging, scale, gap) for image_inputs, image_maps in zip(inputs, maps)]
    return stack_rows(rows, gap)

def feature_maps_image(
    inputs: np.ndarray,
    maps: np.ndarray,
    diverging: bool,
    row_labels: Sequence[str],
    scale: int = 4,
    gap: int = 2,
    section_gap: int = 8,
) -> np.ndarray:
    """
    The grid from feature_maps_grid with its axes labeled, so the image says what each direction means.
    - Across the top, over the input side and the maps side: a title with the tile height and width, then the index of
      each channel.
    - Down the left: one label per row, in `row_labels`, under "image".
    inputs [N, C_in, H_in, W_in], maps [N, C, H, W]. row_labels has N entries.
    """
    image_count, input_channels, input_height, input_width = inputs.shape
    _, map_channels, map_height, map_width = maps.shape
    if len(row_labels) != image_count:
        raise ValueError(f"got {len(row_labels)} row labels for {image_count} images")
    grid = feature_maps_grid(inputs, maps, diverging, scale, gap)
    cell_width = max(input_width, map_width) * scale
    groups = [
        ColumnGroup(f"input ch, {input_height}x{input_width}", [str(index) for index in range(input_channels)], cell_width),
        ColumnGroup(f"output ch, {map_height}x{map_width}", [str(index) for index in range(map_channels)], cell_width),
    ]
    row_height = max(input_height, map_height) * scale
    return add_axis_labels(grid, groups, row_labels, "image", row_height, gap, section_gap)

def log_feature_maps(
    host: dict[str, np.ndarray],
    views: Sequence[FeatureMapView],
    labels: Sequence[str],
    step_number: int,
) -> dict[str, wandb.Image]:
    """
    One labeled image per view: a row per probe image, top to bottom, with the layer's input on the left of each row.
    host: the result of to_cpu_numpy, entries of other monitors are ignored.
    labels: the probe images' labels as text, in row order. They label the rows and go in the caption.
    """
    caption = f"step {step_number}; labels {' '.join(labels)}"
    return {
        f"{NAMESPACE}_{view.layer}/{view.name}": wandb.Image(
            feature_maps_image(
                host[f"{DEVICE_KEY}/{view.layer}/input"],
                host[f"{DEVICE_KEY}/{view.layer}/{view.name}"],
                view.diverging,
                labels,
                view.scale,
            ),
            caption=caption,
        )
        for view in views
    }

def define_feature_map_metrics() -> None:
    wandb.define_metric(METRIC_GLOB, step_metric="step_number")
