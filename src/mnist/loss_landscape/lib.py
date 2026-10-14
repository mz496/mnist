from typing import Literal
from torch import Generator, Tensor, nn, randn

"""
A reimplementation of https://github.com/tomgoldstein/loss-landscape/blob/master/net_plotter.py
"""

type NormalizeIgnore = Literal["bias", "none"]

def get_weights(model: nn.Module) -> list[Tensor]:
    """
    Return the tensor weights of a model.
    """
    return [p.data for p in model.parameters()]

def create_random_weights(weights: list[Tensor], seed: int) -> list[Tensor]:
    """
    From a model's weights, create random weights of the same parameter shapes with the given seed
    """
    generator = Generator().manual_seed(seed)
    return [randn(w.size(), generator=generator).to(w.device) for w in weights]

def create_random_direction(model: nn.Module, seed: int):
    weights: list[Tensor] = get_weights(model)
    direction: list[Tensor] = create_random_weights(weights, seed)
    normalize_directions_for_weights_(direction, weights, ignore="bias")
    return direction

def normalize_directions_for_weights_(direction: list[Tensor], weights: list[Tensor], ignore: NormalizeIgnore) -> None:
    """
    Given a vector d given by `direction`, normalize it in-place so that it is unit-distance relative to some norm of `weights`.
    Optionally ignore bias weights. (Also batch-norm, but we don't have any here)
    """

    # Loop over all params
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == "bias":
                d.fill_(0)
            else:
                d.copy_(w)
        else:
            # Type stubs are bad.
            d.mul_(w.norm() / (d.norm() + 1e-9))

