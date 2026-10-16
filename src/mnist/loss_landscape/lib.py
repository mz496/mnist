import copy
from typing import Literal

import numpy as np
import torch
from torch import Generator, Tensor, nn, randn

"""
Heavily inspired by https://github.com/tomgoldstein/loss-landscape
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

def get_pca_directions_for_snapshots(snapshots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Input: [n, D] array where each row is an observation/snapshot of training weights
    Does SVD to get the top two PCA directions out of V^T (as in U*S*V^T), [2, D]
    Returns:
    - those top two directions [2, D], relative to the snapshots' mean
    - the explained variance ratio of those top two directions [2]
    """
    # Center each row around 0.
    centered = snapshots - snapshots.mean(axis=0)
    # SVD
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    # Top two directions of the PCA.
    components = vt[:2]
    explained_variance_ratio = (s ** 2 / (s ** 2).sum())[:2]
    return components, explained_variance_ratio

def project_snapshots_onto_pca_directions(snapshots: np.ndarray, theta_star: np.ndarray, snapshot_pca_directions: np.ndarray) -> np.ndarray:
    """
    Center each snapshot point (snapshots [n, D]) on theta_star ([D] broadcast to [n, D]) -> [n, D].

    Then dot with PCA components (assumed to be orthonormal) [D, 2] to get per-snapshot points [n, 2].

    Inputs:
    - snapshots: [n, D]
    - theta_star: [D]
    - snapshot_pca_directions: [2, D]

    Output:
    - [n, 2] per-snapshot 2D points
    """
    return (snapshots - theta_star) @ snapshot_pca_directions.T


def unflatten_like(vector: np.ndarray, model: nn.Module) -> list[Tensor]:
    """
    Split a flat [D] vector into one tensor per parameter, shaped and placed like model.parameters().
    Uses vector_to_parameters, the inverse of parameters_to_vector that flattened the training snapshots, so the layer
    order can't drift from how the snapshots were made.
    """
    parameters = list(model.parameters())
    expected = sum(p.numel() for p in parameters)
    # vector_to_parameters ignores extra entries and gives an obscure error for missing ones, so check the length here
    if len(vector) != expected:
        raise ValueError(f"vector has {len(vector)} entries, the model has {expected} parameters")
    scratch = copy.deepcopy(model)
    first = parameters[0]
    flat = torch.from_numpy(vector).to(device=first.device, dtype=first.dtype)  # pyright: ignore[reportUnknownMemberType] Bad stubs from torch.
    torch.nn.utils.vector_to_parameters(flat, scratch.parameters())
    # vector_to_parameters makes the parameters views into `flat`, so clone to hand back independent tensors
    return [p.detach().clone() for p in scratch.parameters()]
