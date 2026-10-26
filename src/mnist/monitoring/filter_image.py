import numpy as np
import torch
import wandb
from matplotlib import colormaps

from mnist.monitoring.image_utils import ColumnGroup, add_axis_labels, stack_rows, tile_row, to_uint8_rgb

# Keys are f"{NAMESPACE}_{layer}/{variant}", so each layer gets its own W&B section.
NAMESPACE = "weights"
METRIC_GLOB = f"{NAMESPACE}_*"
# The prefix of this monitor's tensors in the dict that goes through to_cpu_numpy. The layer name follows the slash.
DEVICE_KEY = "filter_weights"

# Symmetric color limit for the fixed-scale filter image. Default conv1 weights start within +-0.2, so they look
# pale at first, and grow toward this as training takes off. Weights beyond it saturate.
FILTER_VMAX = 1.0

def filters_to_grid(
    weights: np.ndarray, vmax: float | None, scale: int | None = None, gap: int = 2, cell_width: int | None = None
) -> np.ndarray:
    """
    Renders conv filters as upscaled tiles: [F, C_in, K, K] weights -> uint8 RGB [H, W, 3].
    Diverging colormap centered on zero: red is positive, blue is negative, white is zero.
    One column per filter, left to right, and one row per input channel, top to bottom.

    vmax: a fixed symmetric color limit shared by every filter, so magnitudes are comparable across filters and steps.
          None normalizes each filter by its own largest absolute weight, over all of its input channels, which shows
          its pattern but hides its scale.
    scale: each weight is drawn as a scale x scale block of pixels, copied up with nearest-neighbor.
           The default is 24 when there is one input channel and 12 when there are more, which keeps big grids manageable.
           With a cell width the default is the largest whole number of pixels per weight that fits in the cell.
    gap: white pixels between tiles.
    cell_width: give every filter a column this many pixels wide, with its tile centered in it on white. None uses the
                tile's own width. This lets the columns line up with another image. Must be at least a tile wide.
    """
    input_channels, kernel_width = weights.shape[1], weights.shape[3]
    if scale is None:
        scale = max(1, cell_width // kernel_width) if cell_width is not None else (24 if input_channels == 1 else 12)
    if vmax is None:
        limit = np.abs(weights).max(axis=(1, 2, 3), keepdims=True)  # [F, 1, 1, 1]
        # An all-zero filter would divide by zero. Any positive limit renders it as the zero color.
        limit = np.maximum(limit, 1e-12)
    else:
        limit = vmax
    # [-limit, limit] -> [0, 1], with zero at the middle of the colormap
    unit = 0.5 + 0.5 * np.clip(weights / limit, -1.0, 1.0)
    rgb = to_uint8_rgb(colormaps["RdBu_r"](unit))  # [F, C_in, K, K, 3]
    return stack_rows([tile_row(rgb[:, channel], scale, gap, cell_width) for channel in range(input_channels)], gap)

def filters_to_image(
    weights: np.ndarray, vmax: float | None, scale: int | None = None, gap: int = 2, cell_width: int | None = None
) -> np.ndarray:
    """
    The grid from filters_to_grid with its axes labeled, so the image says what each direction means.
    - Across the top: "output ch" and the kernel height and width, then the index of each filter.
    - Down the left: the index of each input channel, under "in ch".
    scale, gap and cell_width are as in filters_to_grid.
    """
    filter_count, input_channels, kernel_height, kernel_width = weights.shape
    if scale is None:
        scale = max(1, cell_width // kernel_width) if cell_width is not None else (24 if input_channels == 1 else 12)
    grid = filters_to_grid(weights, vmax, scale, gap, cell_width)
    column_width = cell_width if cell_width is not None else kernel_width * scale
    columns = ColumnGroup(f"output ch, {kernel_height}x{kernel_width}", [str(index) for index in range(filter_count)], column_width)
    return add_axis_labels(
        grid, [columns], [str(index) for index in range(input_channels)], "in ch", kernel_height * scale, gap, section_gap=0
    )

def collect_filter_weights(layers: dict[str, torch.nn.Conv2d]) -> dict[str, torch.Tensor]:
    """
    The current weights of each conv layer in `layers` (name -> module), as detached device tensors keyed for to_cpu_numpy.
    Raises ValueError for weights that are not [F, C_in, K, K].
    """
    tensors: dict[str, torch.Tensor] = {}
    for name, module in layers.items():
        weight = module.weight
        if weight.dim() != 4:
            raise ValueError(f"{name}: filter images need weights shaped [F, C_in, K, K], got {tuple(weight.shape)}")
        tensors[f"{DEVICE_KEY}/{name}"] = weight.detach()
    return tensors

def log_filter_images(
    host: dict[str, np.ndarray],
    initial: dict[str, np.ndarray],
    step_number: int,
    vmax_by_layer: dict[str, float] | None = None,
) -> dict[str, wandb.Image]:
    """
    Three images per layer:
    - fixed scale: magnitudes comparable across steps
    - normalized: each filter's pattern visible from the start
    - change from init, normalized: only what training has added, without the random starting pattern

    host: the result of to_cpu_numpy, entries of other monitors are ignored.
    initial: the same keys as the filter entries of `host`, taken once before training.
    vmax_by_layer: the fixed color limit per layer name. A layer that is not listed uses FILTER_VMAX.
    Layers differ in weight scale, since the default init shrinks with the number of inputs, so one limit does not suit all.
    Read after optimizer.step(), so these are one update newer than the forward pass the activation stats came from.
    """
    caption = f"step {step_number}"
    vmax_by_layer = vmax_by_layer or {}
    images: dict[str, wandb.Image] = {}
    prefix = f"{DEVICE_KEY}/"
    for key, weights in host.items():
        if not key.startswith(prefix):
            continue
        layer = key[len(prefix):]
        images[f"{NAMESPACE}_{layer}/filters_fixed_scale"] = wandb.Image(filters_to_image(weights, vmax_by_layer.get(layer, FILTER_VMAX)), caption=caption)
        images[f"{NAMESPACE}_{layer}/filters_normalized"] = wandb.Image(filters_to_image(weights, None), caption=caption)
        images[f"{NAMESPACE}_{layer}/filters_change_from_init"] = wandb.Image(
            filters_to_image(weights - initial[key], None), caption=caption
        )
    return images

def define_filter_image_metrics() -> None:
    wandb.define_metric(METRIC_GLOB, step_metric="step_number")
