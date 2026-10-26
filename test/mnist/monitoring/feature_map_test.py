import numpy as np
import torch
from matplotlib import colormaps

from mnist.model import LeNet5
from mnist.monitoring.feature_map import (
    FeatureMapView,
    collect_feature_maps,
    feature_maps_grid,
    feature_maps_image,
    feature_maps_to_image,
    first_example_per_class,
    log_feature_maps,
)
from mnist.monitoring.transfer import to_cpu_numpy


def _conv1_views(model: LeNet5) -> list[FeatureMapView]:
    return [
        FeatureMapView("conv1", "preact", model.conv1, input_from=model.conv1, diverging=True),
        FeatureMapView("conv1", "relu", model.relu1, input_from=model.conv1, diverging=False),
    ]


def test_collect_returns_each_views_output_for_every_image_on_the_device():
    torch.manual_seed(0)
    model = LeNet5()
    images = torch.randn(10, 1, 28, 28)
    tensors = collect_feature_maps(model, images, _conv1_views(model))
    assert set(tensors) == {"feature_maps/conv1/preact", "feature_maps/conv1/relu", "feature_maps/conv1/input"}
    assert all(isinstance(tensor, torch.Tensor) for tensor in tensors.values())
    maps = to_cpu_numpy(tensors)
    assert maps["feature_maps/conv1/preact"].shape == (10, 6, 28, 28)
    assert (maps["feature_maps/conv1/relu"] >= 0).all()
    assert np.allclose(maps["feature_maps/conv1/relu"], np.maximum(maps["feature_maps/conv1/preact"], 0))
    # And it is really conv1's output for these images
    with torch.no_grad():
        expected = model.conv1(images).numpy()
    assert np.allclose(maps["feature_maps/conv1/preact"], expected)
    # The input on the left is the probe images themselves
    assert np.array_equal(maps["feature_maps/conv1/input"], images.numpy())


def test_collect_restores_the_training_mode_and_removes_its_hooks():
    model = LeNet5()
    views = _conv1_views(model)
    hooks_before = (len(model.conv1._forward_hooks), len(model.relu1._forward_hooks))
    image = torch.randn(1, 1, 28, 28)

    model.train()
    collect_feature_maps(model, image, views)
    assert model.training
    model.eval()
    collect_feature_maps(model, image, views)
    assert not model.training
    assert (len(model.conv1._forward_hooks), len(model.relu1._forward_hooks)) == hooks_before


def test_collect_cleans_up_when_the_forward_pass_fails():
    model = LeNet5()
    model.train()
    hooks_before = len(model.conv1._forward_hooks)
    bad_image = torch.randn(1, 3, 28, 28)  # wrong channel count
    try:
        collect_feature_maps(model, bad_image, _conv1_views(model))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the forward pass to fail")
    assert model.training
    assert len(model.conv1._forward_hooks) == hooks_before


def test_capture_does_not_disturb_other_training_hooks():
    model = LeNet5()
    model.train()
    seen: list[int] = []
    # A hook like the ones in train.py: only records in train mode
    model.conv1.register_forward_hook(lambda module, inputs, output: seen.append(1) if module.training else None)
    collect_feature_maps(model, torch.randn(1, 1, 28, 28), _conv1_views(model))
    assert seen == []


def test_image_is_the_input_tiles_then_each_map_in_one_row():
    inputs = np.random.rand(1, 28, 28).astype(np.float32)
    maps = np.random.randn(6, 28, 28).astype(np.float32)
    image = feature_maps_to_image(inputs, maps, diverging=True, scale=4, gap=2, section_gap=8)
    # 1 input tile, a section gap, then 6 map tiles, each 28 * 4 pixels, with 5 ordinary gaps between maps
    assert image.shape == (28 * 4, 28 * 4 + 8 + 6 * 28 * 4 + 5 * 2, 3)
    assert image.dtype == np.uint8
    # The input is grayscale: equal R, G, B
    input_tile = image[:, : 28 * 4]
    assert (input_tile[..., 0] == input_tile[..., 1]).all()
    assert (input_tile[..., 1] == input_tile[..., 2]).all()


def test_both_sides_use_one_scale_and_every_column_is_as_wide_as_the_wider_tile():
    inputs = np.random.rand(6, 14, 14).astype(np.float32)  # 6 input channels of 14 by 14
    maps = np.random.randn(16, 10, 10).astype(np.float32)
    image = feature_maps_to_image(inputs, maps, diverging=True, scale=4, gap=2, section_gap=8)
    # One value is 4 pixels on both sides. The input tiles are 56 wide and 56 tall, the map tiles are 40 by 40,
    # and every column is 56 wide, so the row is 56 tall.
    input_width = 6 * 56 + 5 * 2
    maps_width = 16 * 56 + 15 * 2
    assert image.shape == (56, input_width + 8 + maps_width, 3)
    maps_side = image[:, input_width + 8:]
    # The map tiles are centered vertically, 8 white rows above and below
    assert (maps_side[:8] == 255).all() and (maps_side[-8:] == 255).all()
    # And centered horizontally in each column: 8 white columns on each side of the first and last tile
    assert (maps_side[:, :8] == 255).all()
    assert (maps_side[:, 56 - 8:56] == 255).all()
    assert not (maps_side[8:-8, 8:48] == 255).all()


def test_columns_are_the_same_width_on_both_sides_whichever_tile_is_wider():
    # Here the maps are wider than the inputs, the reverse of the usual case
    inputs = np.random.rand(2, 4, 4).astype(np.float32)
    maps = np.random.randn(3, 6, 6).astype(np.float32)
    image = feature_maps_to_image(inputs, maps, diverging=True, scale=2, gap=1, section_gap=5)
    assert image.shape == (12, (2 * 12 + 1) + 5 + (3 * 12 + 2 * 1), 3)


def test_diverging_maps_put_zero_at_the_center_and_signs_in_opposite_colors():
    inputs = np.zeros((1, 28, 28), dtype=np.float32)
    maps = np.zeros((3, 28, 28), dtype=np.float32)
    maps[1] = 5.0
    maps[2] = -5.0
    image = feature_maps_to_image(inputs, maps, diverging=True, scale=1, gap=0, section_gap=0)
    zero_tile, positive_tile, negative_tile = image[:, 28:56], image[:, 56:84], image[:, 84:112]
    center = (np.asarray(colormaps["RdBu_r"](0.5))[:3] * 255).round().astype(np.uint8)
    assert (zero_tile == center).all()
    assert (positive_tile[..., 0] > positive_tile[..., 2]).all()
    assert (negative_tile[..., 2] > negative_tile[..., 0]).all()


def test_sequential_maps_span_the_colormap_from_each_channels_min_to_max():
    inputs = np.zeros((1, 28, 28), dtype=np.float32)
    maps = np.zeros((1, 28, 28), dtype=np.float32)
    maps[0, 0, 0] = 3.0
    image = feature_maps_to_image(inputs, maps, diverging=False, scale=1, gap=0, section_gap=0)
    tile = image[:, 28:56]
    low = (np.asarray(colormaps["viridis"](0.0))[:3] * 255).round().astype(np.uint8)
    high = (np.asarray(colormaps["viridis"](1.0))[:3] * 255).round().astype(np.uint8)
    assert (tile[1, 1] == low).all()
    assert (tile[0, 0] == high).all()


def test_flat_and_all_zero_inputs_do_not_divide_by_zero():
    inputs = np.zeros((1, 28, 28), dtype=np.float32)
    maps = np.zeros((2, 28, 28), dtype=np.float32)
    for diverging in (True, False):
        image = feature_maps_to_image(inputs, maps, diverging=diverging, scale=1, gap=0, section_gap=0)
        assert np.isfinite(image).all()


def test_grid_stacks_one_row_per_image_with_gaps():
    inputs = np.random.rand(3, 1, 28, 28).astype(np.float32)
    maps = np.random.randn(3, 6, 28, 28).astype(np.float32)
    grid = feature_maps_grid(inputs, maps, diverging=True, scale=2, gap=3)
    row_height = 28 * 2
    row_width = 28 * 2 + 8 + 6 * 28 * 2 + 5 * 3
    assert grid.shape == (3 * row_height + 2 * 3, row_width, 3)
    # The gap between the first two rows is white
    assert (grid[row_height : row_height + 3] == 255).all()
    # And each row matches the single-image rendering of its own image
    for index in range(3):
        expected = feature_maps_to_image(inputs[index], maps[index], diverging=True, scale=2, gap=3)
        top = index * (row_height + 3)
        assert (grid[top : top + row_height] == expected).all()


def _batch(labels: list[int], first_pixel: float) -> tuple[torch.Tensor, torch.Tensor]:
    # Each image is filled with a value that identifies which batch and position it came from
    images = torch.stack([torch.full((1, 28, 28), first_pixel + index) for index in range(len(labels))])
    return images, torch.tensor(labels)


def test_first_example_per_class_keeps_the_first_occurrence_ordered_by_class():
    batches = [_batch([3, 1, 3], 0.0), _batch([0, 1, 2], 10.0)]
    images, labels = first_example_per_class(batches, class_count=4)
    assert labels.tolist() == [0, 1, 2, 3]
    assert images.shape == (4, 1, 28, 28)
    # class 3: first batch, position 0 -> 0.0; class 1: first batch, position 1 -> 1.0
    # class 0: second batch, position 0 -> 10.0; class 2: second batch, position 2 -> 12.0
    assert [float(image[0, 0, 0]) for image in images] == [10.0, 1.0, 12.0, 0.0]


def test_first_example_per_class_stops_scanning_once_every_class_is_found():
    seen: list[int] = []

    def batches():
        for index, labels in enumerate([[0, 1], [2, 3], [0, 1]]):
            seen.append(index)
            yield _batch(labels, 0.0)

    first_example_per_class(batches(), class_count=4)
    assert seen == [0, 1]


def test_first_example_per_class_raises_when_a_class_is_missing():
    try:
        first_example_per_class([_batch([0, 1, 1], 0.0)], class_count=3)
    except ValueError as error:
        assert "[2]" in str(error)
    else:
        raise AssertionError("expected a ValueError")


def test_log_returns_one_image_per_view_named_by_layer_and_view():
    torch.manual_seed(0)
    model = LeNet5()
    views = _conv1_views(model)
    images = torch.randn(3, 1, 28, 28)
    host = to_cpu_numpy(collect_feature_maps(model, images, views))
    logged = log_feature_maps(host, views, labels=["0", "1", "2"], step_number=20)
    assert set(logged) == {"maps_conv1/preact", "maps_conv1/relu"}
    assert logged["maps_conv1/preact"]._caption == "step 20; labels 0 1 2"


def _all_views(model: LeNet5) -> list[FeatureMapView]:
    return _conv1_views(model) + [
        FeatureMapView("conv2", "preact", model.conv2, input_from=model.conv2, diverging=True, scale=6),
        FeatureMapView("conv2", "relu", model.relu2, input_from=model.conv2, diverging=False, scale=6),
    ]


def test_conv2_views_are_collected_as_16_maps_of_10_by_10():
    torch.manual_seed(0)
    model = LeNet5()
    tensors = collect_feature_maps(model, torch.randn(4, 1, 28, 28), _all_views(model))
    maps = to_cpu_numpy(tensors)
    assert maps["feature_maps/conv2/preact"].shape == (4, 16, 10, 10)
    assert (maps["feature_maps/conv2/relu"] >= 0).all()


def test_conv2s_input_is_conv1s_output_after_the_relu_and_the_pool():
    torch.manual_seed(0)
    model = LeNet5()
    images = torch.randn(4, 1, 28, 28)
    maps = to_cpu_numpy(collect_feature_maps(model, images, _all_views(model)))
    conv2_input = maps["feature_maps/conv2/input"]
    assert conv2_input.shape == (4, 6, 14, 14)
    with torch.no_grad():
        expected = torch.nn.functional.avg_pool2d(model.relu1(model.conv1(images)), 2, 2).numpy()
    assert np.allclose(conv2_input, expected)


def test_each_layers_input_is_captured_once_even_with_several_views():
    model = LeNet5()
    tensors = collect_feature_maps(model, torch.randn(2, 1, 28, 28), _all_views(model))
    assert sorted(key for key in tensors if key.endswith("/input")) == ["feature_maps/conv1/input", "feature_maps/conv2/input"]


def test_collect_leaves_no_input_hooks_behind():
    model = LeNet5()
    before = (len(model.conv1._forward_hooks), len(model.conv2._forward_hooks))
    collect_feature_maps(model, torch.randn(2, 1, 28, 28), _all_views(model))
    assert (len(model.conv1._forward_hooks), len(model.conv2._forward_hooks)) == before


def test_log_names_images_by_layer_and_view_and_passes_each_views_scales_to_the_renderer(monkeypatch):
    from mnist.monitoring import feature_map

    calls: list[tuple[int, int]] = []
    real_grid = feature_map.feature_maps_grid

    def recording_grid(inputs, maps, diverging, scale=4, gap=2):
        calls.append(scale)
        return real_grid(inputs, maps, diverging, scale, gap)

    monkeypatch.setattr(feature_map, "feature_maps_grid", recording_grid)
    torch.manual_seed(0)
    model = LeNet5()
    views = _all_views(model)
    images = torch.randn(3, 1, 28, 28)
    host = to_cpu_numpy(collect_feature_maps(model, images, views))
    logged = log_feature_maps(host, views, labels=["0", "1", "2"], step_number=10)
    assert set(logged) == {"maps_conv1/preact", "maps_conv1/relu", "maps_conv2/preact", "maps_conv2/relu"}
    assert logged["maps_conv2/preact"]._caption == "step 10; labels 0 1 2"
    assert calls == [4, 4, 6, 6]


def test_the_labeled_image_frames_the_grid_and_labels_rows_with_the_given_labels():
    inputs = np.random.rand(3, 6, 14, 14).astype(np.float32)
    maps = np.random.randn(3, 16, 10, 10).astype(np.float32)
    grid = feature_maps_grid(inputs, maps, diverging=True, scale=4, gap=2)
    image = feature_maps_image(inputs, maps, diverging=True, row_labels=["7", "8", "9"], scale=4, gap=2)
    assert image.shape == (grid.shape[0] + 26, grid.shape[1] + 40, 3)
    assert (image[26:, 40:] == grid).all()
    assert not (image[26:, :40] == 255).all()


def test_the_labeled_image_needs_one_row_label_per_image():
    inputs = np.random.rand(3, 1, 28, 28).astype(np.float32)
    maps = np.random.randn(3, 6, 28, 28).astype(np.float32)
    try:
        feature_maps_image(inputs, maps, diverging=True, row_labels=["0", "1"])
    except ValueError as error:
        assert "2 row labels" in str(error)
    else:
        raise AssertionError("expected a ValueError")
