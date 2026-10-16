from pathlib import Path
from typing import Any

import numpy as np

"""
Utils to read npz files shared by the loss landscape caches.
"""


def format_tags(**params: Any) -> str:
    """
    Params as name=value tags joined by underscores, for building file names. Underscores in a name become hyphens so
    the name matches its CLI flag: format_tags(mode="2d", max_step=600) is "mode=2d_max-step=600".
    None and False are left out, like a flag that wasn't passed. True is written as "true".
    """
    tags: list[str] = []
    for name, value in params.items():
        if value is None or value is False:
            continue
        tags.append(f"{name.replace('_', '-')}={'true' if value is True else value}")
    return "_".join(tags)


def viz_plots_dir(run_id: str) -> Path:
    """Where a run's static plots go: viz/<run_id>/plots."""
    return Path("viz") / run_id / "plots"


def viz_animations_dir(run_id: str) -> Path:
    """Where a run's animations go: viz/<run_id>/animations."""
    return Path("viz") / run_id / "animations"


def viz_filename(stamp: str, *tags: str, ext: str) -> str:
    """
    A viz file name that starts with the time it was generated, so a folder sorts chronologically, then the tags
    (see format_tags). Empty tags are skipped: viz_filename("20261016-101500", "mode=1d_seed=0", "", ext="png").
    """
    return "_".join([stamp, *(tag for tag in tags if tag)]) + f".{ext}"


def read_arrays(path: Path) -> dict[str, Any]:
    """Every array in an .npz file by key, or an empty dict if the file doesn't exist yet."""
    if not path.is_file():
        return {}
    with np.load(path) as file:
        return {key: file[key] for key in file.files}
