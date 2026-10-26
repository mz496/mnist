from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import wandb

from mnist.monitoring import feature_map, filter_image
from mnist.monitoring.feature_map import FeatureMapView, feature_maps_image
from mnist.monitoring.filter_image import FILTER_VMAX, filters_to_image
from mnist.monitoring.image_utils import GUTTER_WIDTH, join_row, stack_rows

# Keys are f"{NAMESPACE}_{layer}/{view name}", so each layer gets its own W&B section.
NAMESPACE = "layer_view"
METRIC_GLOB = f"{NAMESPACE}_*"

class LayerViewSpec(NamedTuple):
    """
    One combined image: a layer's weights drawn above one of its feature map views.
    - layer: the layer name used by the filter image and feature map monitors. It must be in both.
    - view: the name of the feature map view drawn under the weights.
    """
    layer: str
    view: str

def layer_view_image(
    weights: np.ndarray,
    inputs: np.ndarray,
    maps: np.ndarray,
    diverging: bool,
    row_labels: Sequence[str],
    weights_vmax: float,
    scale: int = 4,
    gap: int = 2,
    section_gap: int = 8,
) -> np.ndarray:
    """
    The labeled weights image stacked above the labeled feature map image, with the output channels lined up.
    Column c holds filter c on top and output channel c of every probe image below.

    weights [F, C_in, K, K], inputs [N, C_in, H_in, W_in], maps [N, F, H, W]. row_labels has N entries.
    The weights tiles get the same cell width as the map tiles, and the weights image is shifted right by the width of
    the map image's input block, so the weights sit directly above the output columns. The block to its left is white.
    weights_vmax: the fixed color limit for the weights. diverging, scale, gap and section_gap are as in feature_maps_image.
    Raises ValueError if the weights and the maps do not have the same number of output channels.
    """
    if weights.shape[0] != maps.shape[1]:
        raise ValueError(f"the weights have {weights.shape[0]} filters but the maps have {maps.shape[1]} output channels")
    cell_width = max(inputs.shape[3], maps.shape[3]) * scale
    weights_image = filters_to_image(weights, weights_vmax, gap=gap, cell_width=cell_width)
    maps_image = feature_maps_image(inputs, maps, diverging, row_labels, scale, gap, section_gap)
    # The maps image is wider by the input block and the section gap. Put that much white after the weights' row labels.
    offset = maps_image.shape[1] - weights_image.shape[1]
    blank = np.full((weights_image.shape[0], offset, 3), 255, dtype=np.uint8)
    aligned = join_row([weights_image[:, :GUTTER_WIDTH], blank, weights_image[:, GUTTER_WIDTH:]], 0)
    return stack_rows([aligned, maps_image], section_gap)

def log_layer_views(
    host: dict[str, np.ndarray],
    specs: Sequence[LayerViewSpec],
    views: Sequence[FeatureMapView],
    labels: Sequence[str],
    step_number: int,
    vmax_by_layer: dict[str, float] | None = None,
) -> dict[str, wandb.Image]:
    """
    One combined image per spec, with the weights above the maps.
    host: the result of to_cpu_numpy. It must hold what the filter image monitor and the feature map monitor collected
          for each spec's layer. Entries of other monitors are ignored.
    views: the feature map views, to find each spec's view and read its scale and colors.
    labels: the probe images' labels as text, in row order. They label the rows and go in the caption.
    vmax_by_layer: the fixed color limit for each layer's weights. A layer that is not listed uses FILTER_VMAX.
    Raises ValueError, naming the layer, if something a spec needs was not collected.
    """
    vmax_by_layer = vmax_by_layer or {}
    caption = f"step {step_number}; labels {' '.join(labels)}"
    images: dict[str, wandb.Image] = {}
    for spec in specs:
        view = next((candidate for candidate in views if (candidate.layer, candidate.name) == (spec.layer, spec.view)), None)
        if view is None:
            raise ValueError(f"layer view {spec.layer}: there is no feature map view named {spec.view}")
        weights_key = f"{filter_image.DEVICE_KEY}/{spec.layer}"
        inputs_key = f"{feature_map.DEVICE_KEY}/{spec.layer}/input"
        maps_key = f"{feature_map.DEVICE_KEY}/{spec.layer}/{spec.view}"
        for key, what in ((weights_key, "filter weights"), (inputs_key, "feature map input"), (maps_key, "feature maps")):
            if key not in host:
                raise ValueError(f"layer view {spec.layer}: no {what} were collected")
        images[f"{NAMESPACE}_{spec.layer}/{spec.view}"] = wandb.Image(
            layer_view_image(
                host[weights_key],
                host[inputs_key],
                host[maps_key],
                view.diverging,
                labels,
                vmax_by_layer.get(spec.layer, FILTER_VMAX),
                view.scale,
            ),
            caption=caption,
        )
    return images

def define_layer_view_metrics() -> None:
    wandb.define_metric(METRIC_GLOB, step_metric="step_number")
