import torch

from mnist.data import MNIST_MEAN, MNIST_STD, collate


def test_collate_normalizes_pixels():
    black = {"image": torch.zeros(1, 28, 28, dtype=torch.uint8), "label": torch.tensor(3)}
    white = {"image": torch.full((1, 28, 28), 255, dtype=torch.uint8), "label": torch.tensor(7)}
    images, labels = collate([black, white])
    assert images.shape == (2, 1, 28, 28)
    assert torch.allclose(images[0], torch.full((1, 28, 28), -MNIST_MEAN / MNIST_STD))
    assert torch.allclose(images[1], torch.full((1, 28, 28), (1 - MNIST_MEAN) / MNIST_STD))
    assert labels.tolist() == [3, 7]
