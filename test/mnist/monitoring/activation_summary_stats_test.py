import torch

from mnist.model import LeNet5
from mnist.monitoring.activation_summary_stats import (
    collect_activation_summary_stats,
    log_activation_summary_stats,
    register_activation_summary_stats,
)
from mnist.monitoring.transfer import to_cpu_numpy


def _layers(model: LeNet5):
    return {"conv1": model.conv1, "relu1": model.relu1, "fc3": model.fc3}


def test_hooks_record_stats_in_train_mode_with_namespaced_keys():
    torch.manual_seed(0)
    model = LeNet5()
    stats, handles = register_activation_summary_stats(_layers(model))
    model.train()
    model(torch.randn(4, 1, 28, 28))
    assert "activation_stats_conv1/mean" in stats
    assert "activation_stats_conv1/std" in stats
    # relu1 follows a convolution, so it also gets the overall and per channel zero fractions
    assert "activation_stats_relu1/frac_zero_mean" in stats
    assert "activation_stats_relu1/frac_zero_ch5" in stats
    assert "activation_stats_fc3/mean" in stats
    assert all(value.dim() == 0 for value in stats.values())
    for handle in handles:
        handle.remove()


def test_hooks_do_not_record_in_eval_mode():
    model = LeNet5()
    stats, _ = register_activation_summary_stats(_layers(model))
    model.eval()
    model(torch.randn(4, 1, 28, 28))
    assert stats == {}


def test_removed_hooks_stop_recording():
    model = LeNet5()
    stats, handles = register_activation_summary_stats(_layers(model))
    for handle in handles:
        handle.remove()
    model.train()
    model(torch.randn(4, 1, 28, 28))
    assert stats == {}


def test_collect_then_log_returns_one_float_per_stat():
    torch.manual_seed(0)
    model = LeNet5()
    stats, _ = register_activation_summary_stats(_layers(model))
    model.train()
    model(torch.randn(4, 1, 28, 28))
    names, device = collect_activation_summary_stats(stats)
    assert len(device) == 1  # one stacked tensor, not one per stat
    logged = log_activation_summary_stats(names, to_cpu_numpy(device))
    assert set(logged) == set(stats)
    assert all(isinstance(value, float) for value in logged.values())
    assert logged["activation_stats_conv1/mean"] == float(stats["activation_stats_conv1/mean"])
