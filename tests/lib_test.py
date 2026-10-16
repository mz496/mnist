import numpy as np
import pytest
import torch
from mnist.model import LeNet5
from mnist.loss_landscape.lib import create_random_direction, get_pca_directions_for_snapshots, plane_base, project_snapshots_onto_pca_directions, unflatten_like


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


def flat_parameters(model: LeNet5) -> np.ndarray:
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy()


def test_unflatten_round_trips_parameters():
    model = LeNet5()
    parts = unflatten_like(flat_parameters(model), model)
    assert len(parts) == 10
    assert all(torch.equal(a, p.detach()) for a, p in zip(parts, model.parameters()))


def test_shifting_by_unflattened_direction_matches_flat_arithmetic():
    # The property the trajectory sweep was missing: a per-layer shift is the same as shifting the flat vector.
    model = LeNet5()
    flat = flat_parameters(model)
    direction = np.random.default_rng(0).standard_normal(flat.shape).astype(np.float32)
    shifted = LeNet5()
    shifted.load_state_dict(model.state_dict())
    with torch.no_grad():
        for p, w, u in zip(shifted.parameters(), [q.detach().clone() for q in model.parameters()], unflatten_like(direction, model)):
            p.copy_(w + 1.7 * u)
    assert np.allclose(flat_parameters(shifted), flat + 1.7 * direction, atol=1e-5)


@pytest.mark.parametrize("length", [5, 61705, 61707])
def test_unflatten_rejects_wrong_length(length: int):
    with pytest.raises(ValueError):
        unflatten_like(np.zeros(length, dtype=np.float32), LeNet5())


def test_plane_base_puts_the_snapshot_at_its_own_point():
    rng = np.random.default_rng(0)
    snapshots = rng.normal(size=(6, 40)).astype(np.float32)
    directions, _ = get_pca_directions_for_snapshots(snapshots)
    points = project_snapshots_onto_pca_directions(snapshots, snapshots[-1], directions)
    for snapshot, point in zip(snapshots, points):
        base = plane_base(snapshot, point, directions)
        # At the snapshot's own (alpha, beta) the plane passes through the snapshot
        assert np.allclose(base + point[0] * directions[0] + point[1] * directions[1], snapshot, atol=1e-5)
        # Anywhere else it is the snapshot moved along the same two directions by the difference in coordinates
        a, b = point[0] + 2.0, point[1] - 0.5
        assert np.allclose(base + a * directions[0] + b * directions[1], snapshot + 2.0 * directions[0] - 0.5 * directions[1], atol=1e-5)


def test_plane_base_for_the_final_snapshot_is_the_final_weights():
    rng = np.random.default_rng(1)
    snapshots = rng.normal(size=(5, 30)).astype(np.float32)
    directions, _ = get_pca_directions_for_snapshots(snapshots)
    points = project_snapshots_onto_pca_directions(snapshots, snapshots[-1], directions)
    # The final snapshot is the center, so it sits at (0, 0) and its plane is the PCA plane itself
    assert np.allclose(points[-1], 0.0, atol=1e-6)
    assert np.allclose(plane_base(snapshots[-1], points[-1], directions), snapshots[-1], atol=1e-6)
