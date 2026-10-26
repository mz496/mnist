import numpy as np
from matplotlib import colormaps

import torch

from mnist.monitoring.filter_image import (
    FILTER_VMAX,
    collect_filter_weights,
    filters_to_grid,
    filters_to_image,
    log_filter_images,
)
from mnist.monitoring.transfer import to_cpu_numpy


def test_image_is_a_row_of_upscaled_tiles_separated_by_gaps():
    weights = np.zeros((6, 1, 5, 5), dtype=np.float32)
    image = filters_to_grid(weights, vmax=1.0, scale=4, gap=1)
    # 5 weights * 4 pixels = 20 per tile; 6 tiles and 5 gaps wide
    assert image.shape == (20, 6 * 20 + 5 * 1, 3)
    assert image.dtype == np.uint8
    # The gap after the first tile is white
    assert (image[:, 20] == 255).all()


def test_zero_weights_render_as_the_colormap_center():
    weights = np.zeros((2, 1, 5, 5), dtype=np.float32)
    center = (np.asarray(colormaps["RdBu_r"](0.5))[:3] * 255).round().astype(np.uint8)
    image = filters_to_grid(weights, vmax=1.0, scale=2, gap=1)
    tile = image[:, :10]
    assert (tile == center).all()


def test_positive_is_red_and_negative_is_blue():
    weights = np.zeros((2, 1, 5, 5), dtype=np.float32)
    weights[0] = 1.0
    weights[1] = -1.0
    image = filters_to_grid(weights, vmax=1.0, scale=1, gap=0)
    positive, negative = image[:, :5], image[:, 5:]
    assert (positive[..., 0] > positive[..., 2]).all()
    assert (negative[..., 2] > negative[..., 0]).all()


def test_fixed_scale_keeps_magnitude_but_normalized_does_not():
    # A tiny filter: pale under a fixed scale, fully saturated when normalized by its own max.
    weights = np.full((1, 1, 5, 5), 0.01, dtype=np.float32)
    fixed = filters_to_grid(weights, vmax=1.0, scale=1, gap=0)
    normalized = filters_to_grid(weights, vmax=None, scale=1, gap=0)
    # Distance from white: bigger means more saturated
    assert (255 - fixed.astype(int)).sum() < (255 - normalized.astype(int)).sum()


def test_normalizing_an_all_zero_filter_does_not_divide_by_zero():
    weights = np.zeros((1, 1, 5, 5), dtype=np.float32)
    image = filters_to_grid(weights, vmax=None, scale=1, gap=0)
    assert np.isfinite(image).all()


def test_collect_returns_detached_device_weights_keyed_by_layer():
    layer = torch.nn.Conv2d(1, 6, 5)
    tensors = collect_filter_weights({"conv1": layer})
    assert list(tensors) == ["filter_weights/conv1"]
    assert tensors["filter_weights/conv1"].shape == (6, 1, 5, 5)
    assert not tensors["filter_weights/conv1"].requires_grad


def test_collect_accepts_layers_with_several_input_channels():
    layer = torch.nn.Conv2d(6, 16, 5)
    tensors = collect_filter_weights({"conv2": layer})
    assert tensors["filter_weights/conv2"].shape == (16, 6, 5, 5)


def test_collect_rejects_weights_that_are_not_4d():
    layer = torch.nn.Linear(10, 4)
    try:
        collect_filter_weights({"fc": layer})  # pyright: ignore[reportArgumentType]
    except ValueError as error:
        assert "fc" in str(error)
    else:
        raise AssertionError("expected a ValueError")


def test_several_input_channels_render_with_a_column_per_filter_and_a_row_per_input_channel():
    weights = np.zeros((16, 6, 5, 5), dtype=np.float32)
    image = filters_to_grid(weights, vmax=1.0, scale=2, gap=1)
    # Tiles are 5 * 2 = 10 pixels. 16 columns and 15 gaps wide, 6 rows and 5 gaps tall.
    assert image.shape == (6 * 10 + 5, 16 * 10 + 15, 3)


def test_the_default_scale_is_smaller_when_there_are_several_input_channels():
    row = filters_to_grid(np.zeros((6, 1, 5, 5), dtype=np.float32), vmax=1.0, gap=0)
    grid = filters_to_grid(np.zeros((16, 6, 5, 5), dtype=np.float32), vmax=1.0, gap=0)
    assert row.shape == (5 * 24, 6 * 5 * 24, 3)
    assert grid.shape == (6 * 5 * 12, 16 * 5 * 12, 3)


def test_filters_are_normalized_over_all_of_their_input_channels():
    # One filter with a large weight in its second input channel and tiny weights in its first.
    weights = np.zeros((1, 2, 5, 5), dtype=np.float32)
    weights[0, 0] = 0.01
    weights[0, 1] = 1.0
    image = filters_to_grid(weights, vmax=None, scale=1, gap=0)
    # Input channels are rows, so the first is on top
    first, second = image[:5], image[5:]
    # They share one scale, so the tiny channel stays pale and the large one is saturated
    assert (255 - first.astype(int)).sum() < (255 - second.astype(int)).sum() / 10


def test_the_labeled_image_frames_the_unlabeled_grid_with_a_header_and_a_left_column():
    weights = np.random.randn(16, 6, 5, 5).astype(np.float32)
    grid = filters_to_grid(weights, vmax=0.5, scale=4, gap=2)
    image = filters_to_image(weights, vmax=0.5, scale=4, gap=2)
    # Two header lines of 13 pixels and a left column of 40
    assert image.shape == (grid.shape[0] + 26, grid.shape[1] + 40, 3)
    assert (image[26:, 40:] == grid).all()
    # The header and the left column hold text, so they are not all white
    assert not (image[:26, 40:] == 255).all()
    assert not (image[26:, :40] == 255).all()


def test_labeled_images_run_output_channels_across_for_one_and_for_several_input_channels():
    one = filters_to_image(np.zeros((6, 1, 5, 5), dtype=np.float32), vmax=1.0, scale=4, gap=2)
    several = filters_to_image(np.zeros((6, 3, 5, 5), dtype=np.float32), vmax=1.0, scale=4, gap=2)
    # The same six filters give the same width, and the extra input channels only add rows
    assert one.shape[1] == several.shape[1]
    assert several.shape[0] == one.shape[0] + 2 * (5 * 4 + 2)


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


def test_each_layer_can_have_its_own_fixed_color_limit():
    layers = {"conv1": torch.nn.Conv2d(1, 6, 5), "conv2": torch.nn.Conv2d(6, 16, 5)}
    initial = to_cpu_numpy(collect_filter_weights(layers))
    host = {key: value + 0.1 for key, value in initial.items()}
    images = log_filter_images(host, initial, step_number=5, vmax_by_layer={"conv2": 0.2})
    assert set(images) == {
        f"weights_{layer}/{variant}"
        for layer in ("conv1", "conv2")
        for variant in ("filters_fixed_scale", "filters_normalized", "filters_change_from_init")
    }
