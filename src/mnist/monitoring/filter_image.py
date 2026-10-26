import numpy as np
import torch
import wandb
from matplotlib import colormaps

from mnist.monitoring.image_utils import tile_row, to_uint8_rgb

# Keys are f"{NAMESPACE}_{layer}/{variant}", so each layer gets its own W&B section.
NAMESPACE = "weights"
METRIC_GLOB = f"{NAMESPACE}_*"
# The prefix of this monitor's tensors in the dict that goes through to_cpu_numpy. The layer name follows the slash.
DEVICE_KEY = "filter_weights"

# Symmetric color limit for the fixed-scale filter image. Default conv1 weights start within +-0.2, so they look
# pale at first, and grow toward this as training takes off. Weights beyond it saturate.
FILTER_VMAX = 1.0

def filters_to_image(weights: np.ndarray, vmax: float | None, scale: int = 24, gap: int = 2) -> np.ndarray:
    """
    Renders conv filters as one row of upscaled tiles: [F, 1, K, K] weights -> uint8 RGB [H, W, 3].
    Diverging colormap centered on zero: red is positive, blue is negative, white is zero.

    vmax: a fixed symmetric color limit shared by every filter, so magnitudes are comparable across filters and steps.
          None normalizes each filter by its own largest absolute weight, which shows its pattern but hides its scale.
    scale: each weight becomes a scale x scale block (nearest-neighbor, so the pixels stay crisp).
    gap: white pixels between tiles.
    """
    filters = weights[:, 0]  # [F, K, K]
    if vmax is None:
        limit = np.abs(filters).max(axis=(1, 2), keepdims=True)  # [F, 1, 1]
        # An all-zero filter would divide by zero. Any positive limit renders it as the zero color.
        limit = np.maximum(limit, 1e-12)
    else:
        limit = vmax
    # [-limit, limit] -> [0, 1], with zero at the middle of the colormap
    unit = 0.5 + 0.5 * np.clip(filters / limit, -1.0, 1.0)
    rgb = to_uint8_rgb(colormaps["RdBu_r"](unit))  # [F, K, K, 3]
    return tile_row(rgb, scale, gap)

def collect_filter_weights(layers: dict[str, torch.nn.Conv2d]) -> dict[str, torch.Tensor]:
    """
    The current weights of each conv layer in `layers` (name -> module), as detached device tensors keyed for to_cpu_numpy.
    Raises ValueError for a layer with more than one input channel, which the renderer does not draw yet.
    """
    tensors: dict[str, torch.Tensor] = {}
    for name, module in layers.items():
        weight = module.weight
        if weight.dim() != 4 or weight.shape[1] != 1:
            raise ValueError(f"{name}: filter images need weights shaped [F, 1, K, K], got {tuple(weight.shape)}")
        tensors[f"{DEVICE_KEY}/{name}"] = weight.detach()
    return tensors

def log_filter_images(host: dict[str, np.ndarray], initial: dict[str, np.ndarray], step_number: int) -> dict[str, wandb.Image]:
    """
    Three images per layer:
    - fixed scale: magnitudes comparable across steps
    - normalized: each filter's pattern visible from the start
    - change from init, normalized: only what training has added, without the random starting pattern

    host: the result of to_cpu_numpy, entries of other monitors are ignored.
    initial: the same keys as the filter entries of `host`, taken once before training.
    Read after optimizer.step(), so these are one update newer than the forward pass the activation stats came from.
    """
    caption = f"step {step_number}"
    images: dict[str, wandb.Image] = {}
    prefix = f"{DEVICE_KEY}/"
    for key, weights in host.items():
        if not key.startswith(prefix):
            continue
        layer = key[len(prefix):]
        images[f"{NAMESPACE}_{layer}/filters_fixed_scale"] = wandb.Image(filters_to_image(weights, FILTER_VMAX), caption=caption)
        images[f"{NAMESPACE}_{layer}/filters_normalized"] = wandb.Image(filters_to_image(weights, None), caption=caption)
        images[f"{NAMESPACE}_{layer}/filters_change_from_init"] = wandb.Image(
            filters_to_image(weights - initial[key], None), caption=caption
        )
    return images

def define_filter_image_metrics() -> None:
    wandb.define_metric(METRIC_GLOB, step_metric="step_number")
