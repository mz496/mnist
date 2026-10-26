import numpy as np
import pytest
import torch

from mnist.model import LeNet5
from mnist.monitoring.feature_map import FeatureMapView, collect_feature_maps, feature_maps_image
from mnist.monitoring.filter_image import collect_filter_weights, filters_to_image
from mnist.monitoring.image_utils import GUTTER_WIDTH, LABEL_HEIGHT
from mnist.monitoring.layer_view import (
    METRIC_GLOB,
    LayerViewSpec,
    layer_view_image,
    log_layer_views,
)
from mnist.monitoring.transfer import to_cpu_numpy


def _views(model: LeNet5) -> list[FeatureMapView]:
    return [
        FeatureMapView("conv1", "preact", model.conv1, input_from=model.conv1, diverging=True),
        FeatureMapView("conv1", "relu", model.relu1, input_from=model.conv1, diverging=False),
        FeatureMapView("conv2", "preact", model.conv2, input_from=model.conv2, diverging=True, scale=5),
        FeatureMapView("conv2", "relu", model.relu2, input_from=model.conv2, diverging=False, scale=5),
    ]


def _host(model: LeNet5, images: torch.Tensor) -> dict[str, np.ndarray]:
    device = {
        **collect_feature_maps(model, images, _views(model)),
        **collect_filter_weights({"conv1": model.conv1, "conv2": model.conv2}),
    }
    return to_cpu_numpy(device)


def _arrays(filters: int = 3, input_channels: int = 2, kernel: int = 2, input_size: int = 4, map_size: int = 4, images: int = 2):
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(filters, input_channels, kernel, kernel)).astype(np.float32)
    inputs = rng.random((images, input_channels, input_size, input_size)).astype(np.float32)
    maps = rng.normal(size=(images, filters, map_size, map_size)).astype(np.float32)
    return weights, inputs, maps


def _columns_with_color(band: np.ndarray) -> np.ndarray:
    """Which pixel columns of an image band hold anything that is not white."""
    return (band != 255).any(axis=(0, 2))


def test_output_columns_of_the_weights_line_up_with_the_output_columns_of_the_maps():
    # Kernel 2 and 4 by 4 inputs and maps at scale 3: a map tile is 12 wide and the weights get 6 pixels per weight,
    # so a weights tile also fills its 12 pixel cell. Then the colored columns of both bands must match exactly.
    weights, inputs, maps = _arrays()
    image = layer_view_image(weights, inputs, maps, True, ["a", "b"], 0.5, scale=3, gap=2, section_gap=8)
    weights_height = filters_to_image(weights, 0.5, gap=2, cell_width=12).shape[0]
    header = 2 * LABEL_HEIGHT
    weights_body = image[header:weights_height, GUTTER_WIDTH:]
    maps_body = image[weights_height + 8 + header:, GUTTER_WIDTH:]
    input_block = 2 * 12 + 2 + 8
    weights_columns = _columns_with_color(weights_body)
    maps_columns = _columns_with_color(maps_body[:, input_block:])
    assert weights_columns[input_block:].tolist() == maps_columns.tolist()
    # Nothing is drawn in the weights band above the input block
    assert not weights_columns[:input_block].any()


def test_the_combined_image_is_as_wide_as_the_maps_image_and_as_tall_as_both_plus_a_gap():
    weights, inputs, maps = _arrays(filters=16, input_channels=6, kernel=5, input_size=14, map_size=10, images=3)
    labels = ["0", "1", "2"]
    image = layer_view_image(weights, inputs, maps, True, labels, 0.5, scale=5, gap=2, section_gap=8)
    maps_image = feature_maps_image(inputs, maps, True, labels, scale=5, gap=2, section_gap=8)
    weights_image = filters_to_image(weights, 0.5, gap=2, cell_width=14 * 5)
    assert image.shape == (weights_image.shape[0] + 8 + maps_image.shape[0], maps_image.shape[1], 3)
    # The maps image is unchanged at the bottom, and the weights keep their own row labels in the left column
    assert (image[-maps_image.shape[0]:] == maps_image).all()
    assert (image[:weights_image.shape[0], :GUTTER_WIDTH] == weights_image[:, :GUTTER_WIDTH]).all()


def test_the_weights_follow_the_map_cell_width():
    # 5 by 5 kernels in a 70 pixel cell get 14 pixels per weight. In a 112 pixel cell they get 22 and sit centered.
    weights = np.ones((2, 1, 5, 5), dtype=np.float32)
    exact = filters_to_image(weights, 1.0, gap=0, cell_width=70)
    assert exact.shape[1] == GUTTER_WIDTH + 2 * 70
    assert exact.shape[0] == 2 * LABEL_HEIGHT + 5 * 14
    narrow = filters_to_image(weights, 1.0, gap=0, cell_width=112)
    body = narrow[2 * LABEL_HEIGHT:, GUTTER_WIDTH:]
    assert body.shape == (5 * 22, 2 * 112, 3)
    # One white pixel on each side of every 110 pixel tile
    assert (body[:, 0] == 255).all() and (body[:, 111] == 255).all()
    assert (body[:, 1:111] != 255).any(axis=0).all()


def test_the_weights_and_maps_must_have_the_same_number_of_output_channels():
    weights, inputs, maps = _arrays(filters=3)
    with pytest.raises(ValueError, match="3 filters"):
        layer_view_image(weights, inputs, maps[:, :2], True, ["a", "b"], 0.5)


def test_log_returns_one_image_per_spec_named_by_layer_and_view():
    torch.manual_seed(0)
    model = LeNet5()
    images = torch.randn(3, 1, 28, 28)
    host = _host(model, images)
    specs = [LayerViewSpec("conv1", "preact"), LayerViewSpec("conv2", "preact")]
    logged = log_layer_views(host, specs, _views(model), ["0", "1", "2"], step_number=40, vmax_by_layer={"conv2": 0.5})
    assert set(logged) == {"layer_view_conv1/preact", "layer_view_conv2/preact"}
    assert logged["layer_view_conv1/preact"]._caption == "step 40; labels 0 1 2"


def test_log_names_the_layer_when_something_it_needs_was_not_collected():
    torch.manual_seed(0)
    model = LeNet5()
    host = _host(model, torch.randn(2, 1, 28, 28))
    views = _views(model)

    without_weights = {key: value for key, value in host.items() if key != "filter_weights/conv2"}
    with pytest.raises(ValueError, match="conv2.*filter weights"):
        log_layer_views(without_weights, [LayerViewSpec("conv2", "preact")], views, ["0", "1"], 10)

    without_maps = {key: value for key, value in host.items() if key != "feature_maps/conv2/preact"}
    with pytest.raises(ValueError, match="conv2.*feature maps"):
        log_layer_views(without_maps, [LayerViewSpec("conv2", "preact")], views, ["0", "1"], 10)

    without_input = {key: value for key, value in host.items() if key != "feature_maps/conv2/input"}
    with pytest.raises(ValueError, match="conv2.*input"):
        log_layer_views(without_input, [LayerViewSpec("conv2", "preact")], views, ["0", "1"], 10)

    with pytest.raises(ValueError, match="conv2.*no feature map view named postnorm"):
        log_layer_views(host, [LayerViewSpec("conv2", "postnorm")], views, ["0", "1"], 10)


def test_log_ignores_entries_that_belong_to_other_monitors():
    torch.manual_seed(0)
    model = LeNet5()
    host = _host(model, torch.randn(2, 1, 28, 28))
    host["activation_stats"] = np.zeros(4, dtype=np.float32)
    logged = log_layer_views(host, [LayerViewSpec("conv1", "preact")], _views(model), ["0", "1"], 10)
    assert list(logged) == ["layer_view_conv1/preact"]


def test_the_metric_glob_matches_only_layer_view_keys():
    assert METRIC_GLOB == "layer_view_*"


def test_a_layer_can_have_a_combined_image_for_each_of_its_views():
    torch.manual_seed(0)
    model = LeNet5()
    host = _host(model, torch.randn(3, 1, 28, 28))
    specs = [
        LayerViewSpec("conv1", "preact"),
        LayerViewSpec("conv1", "relu"),
        LayerViewSpec("conv2", "preact"),
        LayerViewSpec("conv2", "relu"),
    ]
    logged = log_layer_views(host, specs, _views(model), ["0", "1", "2"], step_number=10)
    assert set(logged) == {
        "layer_view_conv1/preact",
        "layer_view_conv1/relu",
        "layer_view_conv2/preact",
        "layer_view_conv2/relu",
    }


def test_a_post_relu_view_is_drawn_on_the_sequential_colormap_under_the_same_weights():
    weights, inputs, maps = _arrays()
    nonnegative = np.abs(maps)
    preact = layer_view_image(weights, inputs, maps, True, ["a", "b"], 0.5, scale=3)
    relu = layer_view_image(weights, inputs, nonnegative, False, ["a", "b"], 0.5, scale=3)
    assert preact.shape == relu.shape
    weights_height = filters_to_image(weights, 0.5, gap=2, cell_width=12).shape[0]
    # The weights band is the same in both, and the maps band is not
    assert (preact[:weights_height] == relu[:weights_height]).all()
    assert not (preact[weights_height:] == relu[weights_height:]).all()
