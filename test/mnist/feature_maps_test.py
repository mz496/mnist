import numpy as np
import torch
from matplotlib import colormaps

from mnist.feature_maps import capture_conv1_maps, feature_maps_grid, feature_maps_to_image, first_example_per_class
from mnist.model import LeNet5


def test_capture_returns_conv1_preactivation_and_relu_for_every_image():
    torch.manual_seed(0)
    model = LeNet5()
    images = torch.randn(10, 1, 28, 28)
    maps = capture_conv1_maps(model, images)
    assert set(maps) == {"conv1_preact", "conv1_relu"}
    assert maps["conv1_preact"].shape == (10, 6, 28, 28)
    assert maps["conv1_relu"].shape == (10, 6, 28, 28)
    assert (maps["conv1_relu"] >= 0).all()
    assert np.allclose(maps["conv1_relu"], np.maximum(maps["conv1_preact"], 0))
    # And it is really conv1's output for these images
    with torch.no_grad():
        expected = model.conv1(images).numpy()
    assert np.allclose(maps["conv1_preact"], expected)


def test_capture_restores_the_training_mode_and_removes_its_hooks():
    model = LeNet5()
    hooks_before = (len(model.conv1._forward_hooks), len(model.relu1._forward_hooks))
    image = torch.randn(1, 1, 28, 28)

    model.train()
    capture_conv1_maps(model, image)
    assert model.training
    model.eval()
    capture_conv1_maps(model, image)
    assert not model.training
    assert (len(model.conv1._forward_hooks), len(model.relu1._forward_hooks)) == hooks_before


def test_capture_cleans_up_when_the_forward_pass_fails():
    model = LeNet5()
    model.train()
    hooks_before = len(model.conv1._forward_hooks)
    bad_image = torch.randn(1, 3, 28, 28)  # wrong channel count
    try:
        capture_conv1_maps(model, bad_image)
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
    capture_conv1_maps(model, torch.randn(1, 1, 28, 28))
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
