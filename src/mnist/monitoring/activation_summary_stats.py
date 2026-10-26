import numpy as np
import torch
import wandb
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

# Keys are f"{NAMESPACE}_{layer}/{stat}", so each layer gets its own W&B section.
NAMESPACE = "activation_stats"
METRIC_GLOB = f"{NAMESPACE}_*"
# The key for this monitor's tensor in the dict that goes through to_cpu_numpy.
DEVICE_KEY = "activation_stats"

def make_stats_hook(name: str, stats: dict[str, torch.Tensor]):
    """
    Returns a forward hook that records stats into `stats` as a dict of key -> 0D tensor.
    - Overwrites on every step.
    - Does not move any values across devices.
    - Only records in train mode.
    """
    @torch.no_grad()
    def hook(module: Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not module.training:
            return
        stats[f"{NAMESPACE}_{name}/mean"] = output.mean()
        stats[f"{NAMESPACE}_{name}/std"] = output.std()
        if isinstance(module, torch.nn.ReLU):
            stats[f"{NAMESPACE}_{name}/frac_zero_mean"] = (output == 0).float().mean()
            # Only look at relus downstream of the convolutions: N, C, H, W.
            if output.dim() == 4:
                zeros = (output == 0).float()
                mean_by_channel = zeros.mean(dim=[0, 2, 3]) # [C]
                for channel in range(mean_by_channel.numel()):
                    stats[f"{NAMESPACE}_{name}/frac_zero_ch{channel}"] = mean_by_channel[channel]
        if isinstance(module, torch.nn.Tanh):
            stats[f"{NAMESPACE}_{name}/frac_saturated"] = (output.abs() > 0.95).float().mean()
            stats[f"{NAMESPACE}_{name}/tanh_grad_mean"] = (1 - output**2).float().mean()
    return hook

def register_activation_summary_stats(layers: dict[str, Module]) -> tuple[dict[str, torch.Tensor], list[RemovableHandle]]:
    """
    Puts a stats hook on every layer in `layers` (name -> module).

    Returns the dict the hooks write into, and the handles to remove the hooks later.
    """
    stats: dict[str, torch.Tensor] = {}
    handles = [module.register_forward_hook(make_stats_hook(name, stats)) for name, module in layers.items()]
    return stats, handles

def collect_activation_summary_stats(stats: dict[str, torch.Tensor]) -> tuple[list[str], dict[str, torch.Tensor]]:
    """
    Stacks the 0D stats into one device tensor, so the transfer moves one tensor instead of about thirty.
    
    Returns the stat keys, in order, and the dict of device tensors for to_cpu_numpy.
    """
    return list(stats), {DEVICE_KEY: torch.stack(list(stats.values()))}

def log_activation_summary_stats(names: list[str], host: dict[str, np.ndarray]) -> dict[str, float]:
    """names: the keys from collect_activation_summary_stats. host: the result of to_cpu_numpy. Returns key -> value."""
    values = host[DEVICE_KEY]
    return {name: float(value) for name, value in zip(names, values)}

def define_activation_summary_stats_metrics() -> None:
    wandb.define_metric(METRIC_GLOB, step_metric="step_number")
