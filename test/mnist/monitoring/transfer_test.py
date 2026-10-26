import numpy as np
import torch

from mnist.monitoring.transfer import to_cpu_numpy


def test_returns_numpy_arrays_with_the_same_shapes_and_values():
    tensors = {"scalar": torch.tensor(3.0), "matrix": torch.arange(6.0).reshape(2, 3)}
    arrays = to_cpu_numpy(tensors)
    assert set(arrays) == {"scalar", "matrix"}
    assert all(isinstance(array, np.ndarray) for array in arrays.values())
    assert arrays["scalar"].shape == ()
    assert arrays["matrix"].shape == (2, 3)
    assert np.array_equal(arrays["matrix"], np.arange(6.0).reshape(2, 3))


def test_results_are_copies_even_for_cpu_tensors():
    # A CPU parameter would otherwise hand back a view of its own memory, which the next optimizer step changes.
    weight = torch.nn.Parameter(torch.ones(3))
    arrays = to_cpu_numpy({"weight": weight})
    with torch.no_grad():
        weight += 1
    assert np.array_equal(arrays["weight"], np.ones(3))


def test_tensors_that_require_grad_are_accepted():
    weight = torch.nn.Parameter(torch.ones(3))
    assert to_cpu_numpy({"weight": weight})["weight"].shape == (3,)
