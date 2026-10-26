import numpy as np
import torch

from mnist.model import LeNet5
from mnist.monitoring.activation_histogram import (
    HISTOGRAM_BINS,
    collect_activation_histograms,
    log_activation_histograms,
    register_activation_histograms,
)
from mnist.monitoring.transfer import to_cpu_numpy


def test_hooks_stash_the_raw_output_of_4d_layers_in_train_mode():
    model = LeNet5()
    stash, _ = register_activation_histograms({"conv1": model.conv1, "conv2": model.conv2})
    model.train()
    model(torch.randn(4, 1, 28, 28))
    assert stash["conv1"].shape == (4, 6, 28, 28)
    assert stash["conv2"].shape == (4, 16, 10, 10)
    assert not stash["conv1"].requires_grad


def test_hooks_skip_outputs_that_are_not_4d():
    model = LeNet5()
    stash, _ = register_activation_histograms({"conv1": model.conv1, "fc1": model.fc1})
    model.train()
    model(torch.randn(4, 1, 28, 28))
    assert set(stash) == {"conv1"}


def test_hooks_do_not_stash_in_eval_mode():
    model = LeNet5()
    stash, _ = register_activation_histograms({"conv1": model.conv1})
    model.eval()
    model(torch.randn(4, 1, 28, 28))
    assert stash == {}


def test_log_returns_one_histogram_per_channel_under_the_preact_namespace():
    torch.manual_seed(0)
    model = LeNet5()
    stash, _ = register_activation_histograms({"conv1": model.conv1, "conv2": model.conv2})
    model.train()
    model(torch.randn(4, 1, 28, 28))
    host = to_cpu_numpy(collect_activation_histograms(stash))
    histograms = log_activation_histograms(host)
    expected = {f"activation_preact_conv1/ch{c}" for c in range(6)} | {f"activation_preact_conv2/ch{c}" for c in range(16)}
    assert set(histograms) == expected
    histogram = histograms["activation_preact_conv1/ch0"]
    assert len(histogram.histogram) == HISTOGRAM_BINS
    # Every value lands in a bin, so the counts add up to images * positions
    assert sum(histogram.histogram) == 4 * 28 * 28


def test_log_takes_the_namespace_as_a_setting_and_ignores_other_monitors_entries():
    host = {"activation_histogram/conv1": np.random.randn(2, 3, 4, 4).astype(np.float32), "activation_stats": np.zeros(5)}
    histograms = log_activation_histograms(host, namespace="activation_postact")
    assert set(histograms) == {f"activation_postact_conv1/ch{c}" for c in range(3)}
