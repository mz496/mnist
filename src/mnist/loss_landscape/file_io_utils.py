from pathlib import Path
from typing import Any

import numpy as np

"""
Utils to read npz files shared by the loss landscape caches.
"""


def read_arrays(path: Path) -> dict[str, Any]:
    """Every array in an .npz file by key, or an empty dict if the file doesn't exist yet."""
    if not path.is_file():
        return {}
    with np.load(path) as file:
        return {key: file[key] for key in file.files}
