from pathlib import Path

import pytest

from mnist.loss_landscape.plot import grid_sweep
from mnist.train import SNAPSHOT_SPACING


@pytest.fixture(autouse=True)
def in_tmp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def test_create_or_load_returns_an_existing_sweep_without_computing(monkeypatch: pytest.MonkeyPatch) -> None:
    path = Path("checkpoints") / "run" / "grid_sweep_mode=1d_seed=0.npz"
    path.parent.mkdir(parents=True)
    path.touch()

    def refuse(*args: object, **kwargs: object) -> Path:
        raise AssertionError("computed a sweep that was already saved")

    monkeypatch.setattr(grid_sweep, "create_grid_sweep", refuse)
    assert grid_sweep.create_or_load_grid_sweep("run", "1d") == path


def test_create_or_load_computes_a_missing_sweep_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_create(run_id: str, mode: str, trajectory: bool, max_step: int | None, seed: int) -> Path:
        calls.append((run_id, mode, trajectory, max_step, seed))
        path = grid_sweep.grid_sweep_path(run_id, mode, trajectory=trajectory, max_step=max_step, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    monkeypatch.setattr(grid_sweep, "create_grid_sweep", fake_create)
    first = grid_sweep.create_or_load_grid_sweep("run", "2d", trajectory=True, max_step=SNAPSHOT_SPACING * 3 + 1)
    again = grid_sweep.create_or_load_grid_sweep("run", "2d", trajectory=True, max_step=SNAPSHOT_SPACING * 3 + 1)
    assert first == again
    # max_step is rounded down to a multiple of the snapshot spacing before the file is looked for, so the second call finds it
    assert calls == [("run", "2d", True, SNAPSHOT_SPACING * 3, 0)]
    assert first.name == f"grid_sweep_mode=2d_trajectory-plot=true_max-step={SNAPSHOT_SPACING * 3}.npz"


def test_create_rejects_trajectory_in_1d() -> None:
    with pytest.raises(ValueError, match="not supported for 1D"):
        grid_sweep.create_grid_sweep("run", "1d", trajectory=True)


def test_create_rejects_max_step_without_trajectory() -> None:
    with pytest.raises(ValueError, match="only applies with trajectory"):
        grid_sweep.create_grid_sweep("run", "2d", max_step=600)
