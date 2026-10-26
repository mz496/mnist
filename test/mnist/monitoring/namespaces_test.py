import fnmatch

import numpy as np
import torch

from mnist.model import LeNet5
from mnist.monitoring import activation_histogram, activation_summary_stats, feature_map, filter_image
from mnist.monitoring.feature_map import FeatureMapView, collect_feature_maps, log_feature_maps
from mnist.monitoring.transfer import to_cpu_numpy


def _logged_keys_by_monitor() -> dict[str, set[str]]:
    """Runs every monitor end to end on a small model, the way train.py does, and returns the keys each one logs."""
    torch.manual_seed(0)
    model = LeNet5()
    stats_layers = {
        "conv1": model.conv1, "relu1": model.relu1, "conv2": model.conv2, "relu2": model.relu2,
        "fc1": model.fc1, "relu3": model.relu3, "fc2": model.fc2, "relu4": model.relu4, "fc3": model.fc3,
    }
    stats, _ = activation_summary_stats.register_activation_summary_stats(stats_layers)
    stash, _ = activation_histogram.register_activation_histograms({"conv1": model.conv1, "conv2": model.conv2})
    filter_layers = {"conv1": model.conv1}
    views = [
        FeatureMapView("conv1", "preact", model.conv1, diverging=True),
        FeatureMapView("conv1", "relu", model.relu1, diverging=False),
    ]
    initial = to_cpu_numpy(filter_image.collect_filter_weights(filter_layers))

    model.train()
    images = torch.randn(10, 1, 28, 28)
    model(images)

    stat_names, stats_device = activation_summary_stats.collect_activation_summary_stats(stats)
    device = {
        **stats_device,
        **activation_histogram.collect_activation_histograms(stash),
        **filter_image.collect_filter_weights(filter_layers),
        **collect_feature_maps(model, images, views),
    }
    host = to_cpu_numpy(device)
    return {
        "stats": set(activation_summary_stats.log_activation_summary_stats(stat_names, host)),
        "histograms": set(activation_histogram.log_activation_histograms(host)),
        "filters": set(filter_image.log_filter_images(host, initial, step_number=10)),
        "maps": set(log_feature_maps(host, views, images[:, 0].numpy(), digits="0", step_number=10)),
    }


def _globs() -> dict[str, str]:
    return {
        "stats": activation_summary_stats.METRIC_GLOB,
        "histograms": activation_histogram.metric_glob(),
        "filters": filter_image.METRIC_GLOB,
        "maps": feature_map.METRIC_GLOB,
    }


def test_every_key_matches_its_own_monitors_glob_and_no_other():
    keys = _logged_keys_by_monitor()
    globs = _globs()
    for monitor, monitor_keys in keys.items():
        assert monitor_keys, f"{monitor} logged nothing"
        for key in monitor_keys:
            matching = [other for other, glob in globs.items() if fnmatch.fnmatchcase(key, glob)]
            assert matching == [monitor], f"{key} matches {matching}, expected only {monitor}"


def test_the_namespaces_are_the_documented_ones():
    keys = _logged_keys_by_monitor()
    assert all(key.startswith("activation_stats_") for key in keys["stats"])
    assert all(key.startswith("activation_preact_") for key in keys["histograms"])
    assert all(key.startswith("weights_") for key in keys["filters"])
    assert all(key.startswith("maps_") for key in keys["maps"])


def test_keys_put_the_layer_in_the_section_before_the_first_slash():
    keys = _logged_keys_by_monitor()
    assert {key.split("/")[0] for key in keys["histograms"]} == {"activation_preact_conv1", "activation_preact_conv2"}
    assert {key.split("/")[0] for key in keys["maps"]} == {"maps_conv1"}
