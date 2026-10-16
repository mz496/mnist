from pathlib import Path
from typing import Any

import numpy as np

"""
Where a grid sweep from compute_surface lives, and how it is written and validated.
A sweep is saved next to the run's checkpoint, as checkpoints/<run_id>/grid_sweep<tag>.npz.
"""


# Version of the loss landscape .npz layout. compute_surface writes it into every file and plot refuses a file whose
# major version differs from this one.
# Bump the minor version when a key is added. Bump the major version when a key is removed, renamed, or changes meaning.
# Version 0.2 keys, for both modes: alphas, loss, acc, seed, run_id, split, mode, schema_version.
# 2d adds betas. 2d with --trajectory-plot adds trajectory and explained_variance.
# 2d with --trajectory-plot and --max-step adds max_step, the last training step in the window.
# 0.2 only added max_step to 0.1, so a reader of one can read the other.
GRID_SWEEP_SCHEMA_VERSION = "0.2"


def grid_sweep_path(run_id: str, mode: str, trajectory: bool = False, max_step: int | None = None) -> Path:
    """
    Sweeps with different parameters get different files, so computing one never overwrites another:
    grid_sweep_1d.npz, grid_sweep_2d.npz, grid_sweep_2d_trajectory.npz, grid_sweep_2d_trajectory_first600.npz
    """
    tag = f"_{mode}"
    if trajectory:
        tag += "_trajectory"
    if max_step is not None:
        tag += f"_first{max_step}"
    return Path("checkpoints") / run_id / f"grid_sweep{tag}.npz"


def save_grid_sweep(path: Path, **arrays: Any) -> None:
    """Write a sweep, adding the schema version so a writer can't forget it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, schema_version=np.array(GRID_SWEEP_SCHEMA_VERSION), **arrays)


def major_version(version: str) -> int:
    """The major part of a "major.minor" version string like "0.1"."""
    return int(version.split(".")[0])


def check_grid_sweep_schema_version(npz: Path, data: np.lib.npyio.NpzFile) -> None:
    """
    Refuse a file that doesn't record its schema version, or whose major version differs from this reader's.
    A different minor version is fine, since a minor bump only adds keys.
    """
    if "schema_version" not in data.files:
        raise ValueError(f"{npz} has no 'schema_version' key. Regenerate it with the current compute_surface, which writes schema {GRID_SWEEP_SCHEMA_VERSION}.")
    file_version = str(data["schema_version"])
    try:
        file_major = major_version(file_version)
    except ValueError:
        raise ValueError(f"{npz} has an unreadable schema_version {file_version!r}, expected something like {GRID_SWEEP_SCHEMA_VERSION!r}.") from None
    reader_major = major_version(GRID_SWEEP_SCHEMA_VERSION)
    if file_major < reader_major:
        raise ValueError(f"{npz} is schema {file_version}, behind this reader's schema {GRID_SWEEP_SCHEMA_VERSION} on the major version. Regenerate it with the current compute_surface.")
    if file_major > reader_major:
        raise ValueError(f"{npz} is schema {file_version}, ahead of this reader's schema {GRID_SWEEP_SCHEMA_VERSION} on the major version. This plot code is out of date for the file. Update it to read that schema.")


def read_str(npz: Path, data: np.lib.npyio.NpzFile, key: str) -> str:
    """A string key that compute_surface always records. Fails loudly if it's missing."""
    if key not in data.files:
        raise ValueError(f"{npz} has no '{key}' key, which compute_surface always records. The file is damaged or was not written by compute_surface.")
    return str(data[key])
