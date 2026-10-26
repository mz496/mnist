import numpy as np

def to_uint8_rgb(rgba: np.ndarray) -> np.ndarray:
    """Drops the alpha channel from colormap output (floats in [0, 1]) and converts to uint8 RGB."""
    return (np.asarray(rgba)[..., :3] * 255).round().astype(np.uint8)

def tile_row(tiles: np.ndarray, scale: int, gap: int) -> np.ndarray:
    """
    Lays [T, h, w, 3] uint8 tiles side by side in one row, each upscaled to (h * scale) x (w * scale)
    by nearest-neighbor, with `gap` white pixels between tiles. Returns uint8 RGB [h * scale, T * w * scale + (T - 1) * gap, 3].
    """
    tiles = tiles.repeat(scale, axis=1).repeat(scale, axis=2)
    spacer = np.full((tiles.shape[1], gap, 3), 255, dtype=np.uint8)
    row: list[np.ndarray] = []
    for index, tile in enumerate(tiles):
        if index > 0:
            row.append(spacer)
        row.append(tile)
    return np.concatenate(row, axis=1)
