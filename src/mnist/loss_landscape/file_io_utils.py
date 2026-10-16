from pathlib import Path
from typing import Any

import numpy as np

"""
Utils to read and write files to npz for passing between loss landscape computation and plotting.
Ideally also encompasses load() and save().
"""


# Version of the loss landscape .npz layout. compute_surface writes it into every file and plot refuses a file whose
# major version differs from this one.
# Bump the minor version when a key is added. Bump the major version when a key is removed, renamed, or changes meaning.
# Version 0.2 keys, for both modes: alphas, loss, acc, seed, run_id, split, mode, schema_version.
# 2d adds betas. 2d with --trajectory-plot adds trajectory and explained_variance.
# 2d with --trajectory-plot and --max-step adds max_step, the last training step in the window.
# 0.2 only added max_step to 0.1, so a reader of one can read the other.
SCHEMA_VERSION = "0.2"


def major_version(version: str) -> int:
    """The major part of a "major.minor" version string like "0.1"."""
    return int(version.split(".")[0])


def check_schema_version(npz: Path, data: np.lib.npyio.NpzFile) -> None:
    """
    Refuse a file that doesn't record its schema version, or whose major version differs from this reader's.
    A different minor version is fine, since a minor bump only adds keys.
    """
    if "schema_version" not in data.files:
        raise ValueError(f"{npz} has no 'schema_version' key. Regenerate it with the current compute_surface, which writes schema {SCHEMA_VERSION}.")
    file_version = str(data["schema_version"])
    try:
        file_major = major_version(file_version)
    except ValueError:
        raise ValueError(f"{npz} has an unreadable schema_version {file_version!r}, expected something like {SCHEMA_VERSION!r}.") from None
    reader_major = major_version(SCHEMA_VERSION)
    if file_major < reader_major:
        raise ValueError(f"{npz} is schema {file_version}, behind this reader's schema {SCHEMA_VERSION} on the major version. Regenerate it with the current compute_surface.")
    if file_major > reader_major:
        raise ValueError(f"{npz} is schema {file_version}, ahead of this reader's schema {SCHEMA_VERSION} on the major version. This plot code is out of date for the file. Update it to read that schema.")


def read_str(npz: Path, data: np.lib.npyio.NpzFile, key: str) -> str:
    """A string key that compute_surface always records. Fails loudly if it's missing."""
    if key not in data.files:
        raise ValueError(f"{npz} has no '{key}' key, which compute_surface always records. The file is damaged or was not written by compute_surface.")
    return str(data[key])


def read_arrays(path: Path) -> dict[str, Any]:
    """Every array in an .npz file by key, or an empty dict if the file doesn't exist yet."""
    if not path.is_file():
        return {}
    with np.load(path) as file:
        return {key: file[key] for key in file.files}
