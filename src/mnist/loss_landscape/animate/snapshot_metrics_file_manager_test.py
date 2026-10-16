from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist.loss_landscape.animate import snapshot_metrics_file_manager as snapshot_metrics
from mnist.loss_landscape.animate.snapshot_metrics_file_manager import create_or_load_snapshot_metrics
from mnist.model import LeNet5


def test_evaluating_snapshots_runs_the_network_and_reuses_the_saved_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    parameter_count = sum(p.numel() for p in LeNet5().parameters())
    generator = np.random.default_rng(0)
    snapshots = generator.normal(0.0, 0.05, size=(4, parameter_count)).astype(np.float32)
    batch = (torch.rand(6, 1, 28, 28), torch.tensor([0, 1, 2, 3, 4, 5]))

    class FakeLoaders:
        train_subset = [batch]
        validation = [batch]

    def fake_snapshots(run_id: str) -> np.ndarray:
        return snapshots

    def fake_loaders(data_config: object) -> FakeLoaders:
        return FakeLoaders()

    # The working directory is the temporary one, where there is no configs/default.yaml to read
    monkeypatch.setattr(snapshot_metrics, "load_default_config", lambda: SimpleNamespace(data=None))
    monkeypatch.setattr(snapshot_metrics, "load_training_snapshots", fake_snapshots)
    monkeypatch.setattr(snapshot_metrics, "get_dataloaders", fake_loaders)
    monkeypatch.setattr(snapshot_metrics, "get_device", lambda: "cpu")

    first = create_or_load_snapshot_metrics("run", "train", 4)
    assert first.loss.shape == (4,) and first.acc.shape == (4,)
    assert np.all(np.isfinite(first.loss)) and np.all((first.acc >= 0) & (first.acc <= 1))
    # Different weights give different losses, which shows each snapshot is really loaded into the network
    assert len(set(np.round(first.loss, 6))) == 4
    # Saved next to the snapshots, in the file named like training_snapshot_vectors.npz
    assert snapshot_metrics.metrics_path("run") == Path("checkpoints") / "run" / "training_snapshot_metrics.npz"
    assert snapshot_metrics.metrics_path("run").is_file()

    # A second call is answered from the saved file: running the network again would raise
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the network was evaluated again")

    monkeypatch.setattr(snapshot_metrics, "evaluate", refuse)
    again = create_or_load_snapshot_metrics("run", "train", 4)
    assert np.array_equal(again.loss, first.loss)
    assert np.array_equal(create_or_load_snapshot_metrics("run", "train", 2).loss, first.loss[:2])


def test_each_split_keeps_its_own_results_in_the_one_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    parameter_count = sum(p.numel() for p in LeNet5().parameters())
    snapshots = np.random.default_rng(1).normal(0.0, 0.05, size=(3, parameter_count)).astype(np.float32)
    train_batch = (torch.rand(6, 1, 28, 28), torch.tensor([0, 1, 2, 3, 4, 5]))
    validation_batch = (torch.rand(6, 1, 28, 28), torch.tensor([5, 4, 3, 2, 1, 0]))

    class FakeLoaders:
        train_subset = [train_batch]
        validation = [validation_batch]

    def fake_snapshots(run_id: str) -> np.ndarray:
        return snapshots

    def fake_loaders(data_config: object) -> FakeLoaders:
        return FakeLoaders()

    monkeypatch.setattr(snapshot_metrics, "load_default_config", lambda: SimpleNamespace(data=None))
    monkeypatch.setattr(snapshot_metrics, "load_training_snapshots", fake_snapshots)
    monkeypatch.setattr(snapshot_metrics, "get_dataloaders", fake_loaders)
    monkeypatch.setattr(snapshot_metrics, "get_device", lambda: "cpu")

    train = create_or_load_snapshot_metrics("run", "train", 3)
    validation = create_or_load_snapshot_metrics("run", "validation", 3)
    with np.load(snapshot_metrics.metrics_path("run")) as saved:
        assert sorted(saved.files) == ["train_acc", "train_loss", "validation_acc", "validation_loss"]
        # Measuring validation did not discard the train results
        assert np.array_equal(saved["train_loss"], train.loss)
        assert np.array_equal(saved["validation_loss"], validation.loss)
