from pathlib import Path

import numpy as np
import pytest

from mnist.loss_landscape.file_io_utils import format_tags, viz_animations_dir, viz_filename, viz_plots_dir
from mnist.loss_landscape.grid_sweep_file_manager import GRID_SWEEP_SCHEMA_VERSION, check_grid_sweep_schema_version, grid_sweep_path, save_grid_sweep, sweep_tags


def test_format_tags_names_match_cli_flags_and_skip_unset_params() -> None:
    assert format_tags(mode="2d", trajectory_plot=True, max_step=600) == "mode=2d_trajectory-plot=true_max-step=600"
    assert format_tags(mode="1d", trajectory_plot=False, max_step=None, seed=0) == "mode=1d_seed=0"


def test_grid_sweep_path_is_tagged_by_params_and_lives_with_the_checkpoint() -> None:
    base = Path("checkpoints") / "run"
    assert grid_sweep_path("run", "1d") == base / "grid_sweep_mode=1d_seed=0.npz"
    assert grid_sweep_path("run", "2d", seed=3) == base / "grid_sweep_mode=2d_seed=3.npz"
    # A trajectory sweep ignores the seed, so the seed isn't part of its name
    assert grid_sweep_path("run", "2d", trajectory=True, seed=3) == base / "grid_sweep_mode=2d_trajectory-plot=true.npz"
    assert grid_sweep_path("run", "2d", trajectory=True, max_step=600) == base / "grid_sweep_mode=2d_trajectory-plot=true_max-step=600.npz"


def test_save_grid_sweep_records_the_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "run" / "grid_sweep.npz"
    save_grid_sweep(path, alphas=np.zeros(3))
    with np.load(path) as data:
        assert str(data["schema_version"]) == GRID_SWEEP_SCHEMA_VERSION
        check_grid_sweep_schema_version(path, data)


@pytest.mark.parametrize("version", ["not-a-version", "-1.0", "99.0"])
def test_check_grid_sweep_schema_version_rejects_bad_versions(tmp_path: Path, version: str) -> None:
    path = tmp_path / "sweep.npz"
    np.savez(path, schema_version=np.array(version))
    with np.load(path) as data, pytest.raises(ValueError):
        check_grid_sweep_schema_version(path, data)


def test_check_grid_sweep_schema_version_rejects_missing_version(tmp_path: Path) -> None:
    path = tmp_path / "sweep.npz"
    np.savez(path, alphas=np.zeros(1))
    with np.load(path) as data, pytest.raises(ValueError, match="no 'schema_version'"):
        check_grid_sweep_schema_version(path, data)


def test_viz_dirs_are_namespaced_by_run_id() -> None:
    assert viz_plots_dir("run") == Path("viz") / "run" / "plots"
    assert viz_animations_dir("run") == Path("viz") / "run" / "animations"


def test_viz_filename_starts_with_the_stamp_and_skips_empty_tags() -> None:
    assert viz_filename("20261016-101500", "mode=2d", "value=loss", ext="png") == "20261016-101500_mode=2d_value=loss.png"
    assert viz_filename("20261016-101500", "mode=1d_seed=0", "", ext="png") == "20261016-101500_mode=1d_seed=0.png"


def test_sweep_tags_strip_the_file_prefix() -> None:
    assert sweep_tags(grid_sweep_path("run", "2d", trajectory=True, max_step=600)) == "mode=2d_trajectory-plot=true_max-step=600"
