import numpy as np
import torch
import wandb
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

# Keys are f"{namespace}_{layer}/ch{c}", so each layer gets its own W&B section.
# The namespace is a setting so a second histogram of post-activation values can get its own prefix later.
NAMESPACE = "activation_preact"
# The prefix of this monitor's tensors in the dict that goes through to_cpu_numpy. The layer name follows the slash.
DEVICE_KEY = "activation_histogram"
# Bins per W&B histogram (default 64, max 512). Fine bins help resolve the thin background spike in conv1.
HISTOGRAM_BINS = 256

def metric_glob(namespace: str = NAMESPACE) -> str:
    return f"{namespace}_*"

def make_stash_hook(name: str, stash: dict[str, torch.Tensor]):
    """
    Returns a forward hook that stashes the module's raw output in `stash[name]`.
    - Overwrites on every step.
    - Stores a reference only: no copy, no device transfer.
    - Only records in train mode.
    - Skips outputs that are not 4D, since a histogram per channel needs [N, C, H, W].
    """
    def hook(module: Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not module.training or output.dim() != 4:
            return
        # no_grad would not alter `output`, which still carries its grad_fn. We hold the tensor itself, so detach it
        # to avoid keeping this step's autograd graph alive.
        stash[name] = output.detach()
    return hook

def register_activation_histograms(layers: dict[str, Module]) -> tuple[dict[str, torch.Tensor], list[RemovableHandle]]:
    """
    Puts a stash hook on every layer in `layers` (name -> module).
    Returns the dict the hooks write into, and the handles to remove the hooks later.
    """
    stash: dict[str, torch.Tensor] = {}
    handles = [module.register_forward_hook(make_stash_hook(name, stash)) for name, module in layers.items()]
    return stash, handles

def collect_activation_histograms(stash: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The stashed device tensors, keyed for to_cpu_numpy."""
    return {f"{DEVICE_KEY}/{layer}": output for layer, output in stash.items()}

def log_activation_histograms(host: dict[str, np.ndarray], namespace: str = NAMESPACE) -> dict[str, wandb.Histogram]:
    """
    One W&B histogram per channel of each stashed [N, C, H, W] array, over every image and position.
    The bin range is each channel's own min/max at that step, so bin edges differ between steps.
    host: the result of to_cpu_numpy. Entries that belong to other monitors are ignored.
    """
    histograms: dict[str, wandb.Histogram] = {}
    prefix = f"{DEVICE_KEY}/"
    for key, output in host.items():
        if not key.startswith(prefix):
            continue
        layer = key[len(prefix):]
        for channel in range(output.shape[1]):
            histograms[f"{namespace}_{layer}/ch{channel}"] = wandb.Histogram(
                np_histogram=np.histogram(output[:, channel], bins=HISTOGRAM_BINS)
            )
    return histograms

def define_activation_histogram_metrics(namespace: str = NAMESPACE) -> None:
    wandb.define_metric(metric_glob(namespace), step_metric="step_number")
