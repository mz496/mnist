import torch
from mnist.model import LeNet5
from mnist.loss_landscape.lib import create_random_direction


def directions_equal(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_same_seed_gives_same_direction():
    # Same model, so the normalization (which depends on the weights) is identical too.
    model = LeNet5()
    assert directions_equal(create_random_direction(model, 0), create_random_direction(model, 0))


def test_different_seeds_give_different_directions():
    model = LeNet5()
    assert not directions_equal(create_random_direction(model, 0), create_random_direction(model, 1))


def test_direction_does_not_touch_global_rng():
    model = LeNet5()
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    create_random_direction(model, 0)
    assert torch.equal(torch.rand(3), expected)
