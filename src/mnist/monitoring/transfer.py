import numpy as np
import torch

def to_cpu_numpy(tensors: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """
    The one place in the monitoring package where data leaves the device.
    Every monitor's collect step returns device tensors, and every monitor's log step takes the NumPy arrays this returns.

    Each result is a copy, even when the tensor is already on the CPU. Otherwise a CPU parameter would hand back a view
    of its own memory, and the next optimizer step would change an array we meant to keep.

    NOTE: could also do a single copy instead of one per tensor:
    - Flatten every tensor with reshape(-1) and torch.cat them into one 1-D tensor on the device.
    - Call .cpu() once on that tensor.
    - Split the result back by the recorded sizes and reshape each piece to its original shape.
    This needs one dtype across all tensors. It trades a device-side concatenation and a temporary of the total size
    for one blocking copy instead of one per tensor. At one logging step in ten the saving is about a millisecond,
    so the simple loop stays until that shows up in a profile.
    """
    return {name: tensor.detach().to("cpu", copy=True).numpy() for name, tensor in tensors.items()}
