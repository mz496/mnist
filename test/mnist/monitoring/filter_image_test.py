import numpy as np
from matplotlib import colormaps

import torch

from mnist.monitoring.filter_image import (
    FILTER_VMAX,
    collect_filter_weights,
    filters_to_image,
    log_filter_images,
)
from mnist.monitoring.transfer import to_cpu_numpy


def test_image_is_a_row_of_upscaled_tiles_separated_by_gaps():
    weights = np.zeros((6, 1, 5, 5), dtype=np.float32)
    image = filters_to_image(weights, vmax=1.0, scale=4, gap=1)
    # 5 weights * 4 pixels = 20 per tile; 6 tiles and 5 gaps wide
    assert image.shape == (20, 6 * 20 + 5 * 1, 3)
    assert image.dtype == np.uint8
    # The gap after the first tile is white
    assert (image[:, 20] == 255).all()


def test_zero_weights_render_as_the_colormap_center():
    weights = np.zeros((2, 1, 5, 5), dtype=np.float32)
    center = (np.asarray(colormaps["RdBu_r"](0.5))[:3] * 255).round().astype(np.uint8)
    image = filters_to_image(weights, vmax=1.0, scale=2, gap=1)
    tile = image[:, :10]
    assert (tile == center).all()


def test_positive_is_red_and_negative_is_blue():
    weights = np.zeros((2, 1, 5, 5), dtype=np.float32)
    weights[0] = 1.0
    weights[1] = -1.0
    image = filters_to_image(weights, vmax=1.0, scale=1, gap=0)
    positive, negative = image[:, :5], image[:, 5:]
    assert (positive[..., 0] > positive[..., 2]).all()
    assert (negative[..., 2] > negative[..., 0]).all()


def test_fixed_scale_keeps_magnitude_but_normalized_does_not():
    # A tiny filter: pale under a fixed scale, fully saturated when normalized by its own max.
    weights = np.full((1, 1, 5, 5), 0.01, dtype=np.float32)
    fixed = filters_to_image(weights, vmax=1.0, scale=1, gap=0)
    normalized = filters_to_image(weights, vmax=None, scale=1, gap=0)
    # Distance from white: bigger means more saturated
    assert (255 - fixed.astype(int)).sum() < (255 - normalized.astype(int)).sum()


def test_normalizing_an_all_zero_filter_does_not_divide_by_zero():
    weights = np.zeros((1, 1, 5, 5), dtype=np.float32)
    image = filters_to_image(weights, vmax=None, scale=1, gap=0)
    assert np.isfinite(image).all()


def test_collect_returns_detached_device_weights_keyed_by_layer():
    layer = torch.nn.Conv2d(1, 6, 5)
    tensors = collect_filter_weights({"conv1": layer})
    assert list(tensors) == ["filter_weights/conv1"]
    assert tensors["filter_weights/conv1"].shape == (6, 1, 5, 5)
    assert not tensors["filter_weights/conv1"].requires_grad


def test_collect_rejects_layers_with_more_than_one_input_channel():
    layer = torch.nn.Conv2d(6, 16, 5)
    try:
        collect_filter_weights({"conv2": layer})
    except ValueError as error:
        assert "conv2" in str(error)
    else:
        raise AssertionError("expected a ValueError")


def test_log_returns_three_images_per_layer_with_the_step_in_the_caption():
    layer = torch.nn.Conv2d(1, 6, 5)
    initial = to_cpu_numpy(collect_filter_weights({"conv1": layer}))
    with torch.no_grad():
        layer.weight += 0.1
    host = to_cpu_numpy(collect_filter_weights({"conv1": layer}))
    images = log_filter_images(host, initial, step_number=30)
    assert set(images) == {
        "weights_conv1/filters_fixed_scale",
        "weights_conv1/filters_normalized",
        "weights_conv1/filters_change_from_init",
    }
    assert images["weights_conv1/filters_fixed_scale"]._caption == "step 30"


def test_log_ignores_entries_that_belong_to_other_monitors():
    layer = torch.nn.Conv2d(1, 6, 5)
    host = to_cpu_numpy({**collect_filter_weights({"conv1": layer}), "activation_stats": torch.zeros(3)})
    initial = {k: v for k, v in host.items() if k.startswith("filter_weights/")}
    images = log_filter_images(host, initial, step_number=1)
    assert len(images) == 3


def test_the_fixed_scale_limit_is_a_positive_constant():
    assert FILTER_VMAX > 0
