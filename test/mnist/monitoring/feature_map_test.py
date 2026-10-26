import numpy as np
import torch
from matplotlib import colormaps

from mnist.model import LeNet5
from mnist.monitoring.feature_map import (
    FeatureMapView,
    collect_feature_maps,
    feature_maps_grid,
    feature_maps_to_image,
    first_example_per_class,
    log_feature_maps,
)
from mnist.monitoring.transfer import to_cpu_numpy


def _conv1_views(model: LeNet5) -> list[FeatureMapView]:
    return [
        FeatureMapView("conv1", "preact", model.conv1, diverging=True),
        FeatureMapView("conv1", "relu", model.relu1, diverging=False),
    ]


def test_collect_returns_each_views_output_for_every_image_on_the_device():
    torch.manual_seed(0)
    model = LeNet5()
    images = torch.randn(10, 1, 28, 28)
    tensors = collect_feature_maps(model, images, _conv1_views(model))
    assert set(tensors) == {"feature_maps/conv1/preact", "feature_maps/conv1/relu"}
    assert all(isinstance(tensor, torch.Tensor) for tensor in tensors.values())
    maps = to_cpu_numpy(tensors)
    assert maps["feature_maps/conv1/preact"].shape == (10, 6, 28, 28)
    assert (maps["feature_maps/conv1/relu"] >= 0).all()
    assert np.allclose(maps["feature_maps/conv1/relu"], np.maximum(maps["feature_maps/conv1/preact"], 0))
    # And it is really conv1's output for these images
    with torch.no_grad():
        expected = model.conv1(images).numpy()
    assert np.allclose(maps["feature_maps/conv1/preact"], expected)


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


def test_image_is_the_original_then_each_map_in_one_row():
    original = np.random.rand(28, 28).astype(np.float32)
    maps = np.random.randn(6, 28, 28).astype(np.float32)
    image = feature_maps_to_image(original, maps, diverging=True, scale=4, gap=2)
    # 7 tiles (original + 6 maps) of 28 * 4 pixels, 6 gaps of 2
    assert image.shape == (28 * 4, 7 * 28 * 4 + 6 * 2, 3)
    assert image.dtype == np.uint8
    # The original is grayscale: equal R, G, B
    original_tile = image[:, : 28 * 4]
    assert (original_tile[..., 0] == original_tile[..., 1]).all()
    assert (original_tile[..., 1] == original_tile[..., 2]).all()


def test_diverging_maps_put_zero_at_the_center_and_signs_in_opposite_colors():
    original = np.zeros((28, 28), dtype=np.float32)
    maps = np.zeros((3, 28, 28), dtype=np.float32)
    maps[1] = 5.0
    maps[2] = -5.0
    image = feature_maps_to_image(original, maps, diverging=True, scale=1, gap=0)
    zero_tile, positive_tile, negative_tile = image[:, 28:56], image[:, 56:84], image[:, 84:112]
    center = (np.asarray(colormaps["RdBu_r"](0.5))[:3] * 255).round().astype(np.uint8)
    assert (zero_tile == center).all()
    assert (positive_tile[..., 0] > positive_tile[..., 2]).all()
    assert (negative_tile[..., 2] > negative_tile[..., 0]).all()


def test_sequential_maps_span_the_colormap_from_each_channels_min_to_max():
    original = np.zeros((28, 28), dtype=np.float32)
    maps = np.zeros((1, 28, 28), dtype=np.float32)
    maps[0, 0, 0] = 3.0
    image = feature_maps_to_image(original, maps, diverging=False, scale=1, gap=0)
    tile = image[:, 28:56]
    low = (np.asarray(colormaps["viridis"](0.0))[:3] * 255).round().astype(np.uint8)
    high = (np.asarray(colormaps["viridis"](1.0))[:3] * 255).round().astype(np.uint8)
    assert (tile[1, 1] == low).all()
    assert (tile[0, 0] == high).all()


def test_flat_and_all_zero_inputs_do_not_divide_by_zero():
    original = np.zeros((28, 28), dtype=np.float32)
    maps = np.zeros((2, 28, 28), dtype=np.float32)
    for diverging in (True, False):
        image = feature_maps_to_image(original, maps, diverging=diverging, scale=1, gap=0)
        assert np.isfinite(image).all()


def test_grid_stacks_one_row_per_image_with_gaps():
    originals = np.random.rand(3, 28, 28).astype(np.float32)
    maps = np.random.randn(3, 6, 28, 28).astype(np.float32)
    grid = feature_maps_grid(originals, maps, diverging=True, scale=2, gap=3)
    row_height = 28 * 2
    assert grid.shape == (3 * row_height + 2 * 3, 7 * 28 * 2 + 6 * 3, 3)
    # The gap between the first two rows is white
    assert (grid[row_height : row_height + 3] == 255).all()
    # And each row matches the single-row rendering of its own image
    for index in range(3):
        expected = feature_maps_to_image(originals[index], maps[index], diverging=True, scale=2, gap=3)
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
    logged = log_feature_maps(host, views, images[:, 0].numpy(), digits="0 1 2", step_number=20)
    assert set(logged) == {"maps_conv1/preact", "maps_conv1/relu"}
    assert logged["maps_conv1/preact"]._caption == "step 20; digits 0 1 2"
