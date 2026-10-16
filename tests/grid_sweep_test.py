from pathlib import Path

import numpy as np
import pytest

from mnist.loss_landscape.grid_sweep_file_manager import SCHEMA_VERSION, check_sweep_schema_version, grid_sweep_path, save_grid_sweep


def test_grid_sweep_path_is_tagged_by_params_and_lives_with_the_checkpoint() -> None:
    base = Path("checkpoints") / "run"
    assert grid_sweep_path("run", "1d") == base / "grid_sweep_1d.npz"
    assert grid_sweep_path("run", "2d") == base / "grid_sweep_2d.npz"
    assert grid_sweep_path("run", "2d", trajectory=True) == base / "grid_sweep_2d_trajectory.npz"
    assert grid_sweep_path("run", "2d", trajectory=True, max_step=600) == base / "grid_sweep_2d_trajectory_first600.npz"


def test_save_grid_sweep_records_the_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "run" / "grid_sweep_1d.npz"
    save_grid_sweep(path, alphas=np.zeros(3))
    with np.load(path) as data:
        assert str(data["schema_version"]) == SCHEMA_VERSION
        check_sweep_schema_version(path, data)


@pytest.mark.parametrize("version", ["not-a-version", "-1.0", "99.0"])
def test_check_sweep_schema_version_rejects_bad_versions(tmp_path: Path, version: str) -> None:
    path = tmp_path / "sweep.npz"
    np.savez(path, schema_version=np.array(version))
    with np.load(path) as data, pytest.raises(ValueError):
        check_sweep_schema_version(path, data)


def test_check_sweep_schema_version_rejects_missing_version(tmp_path: Path) -> None:
    path = tmp_path / "sweep.npz"
    np.savez(path, alphas=np.zeros(1))
    with np.load(path) as data, pytest.raises(ValueError, match="no 'schema_version'"):
        check_sweep_schema_version(path, data)
