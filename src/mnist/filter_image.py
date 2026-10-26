import numpy as np
from matplotlib import colormaps

# Symmetric color limit for the fixed-scale filter image. Default conv1 weights start within +-0.2, so they look
# pale at first, and grow toward this as training takes off. Weights beyond it saturate.
FILTER_VMAX = 1.0

def filters_to_image(weights: np.ndarray, vmax: float | None, scale: int = 24, gap: int = 2) -> np.ndarray:
    """
    Renders conv filters as one row of upscaled tiles: [F, 1, K, K] weights -> uint8 RGB [H, W, 3].
    Diverging colormap centered on zero: red is positive, blue is negative, white is zero.

    vmax: a fixed symmetric color limit shared by every filter, so magnitudes are comparable across filters and steps.
          None normalizes each filter by its own largest absolute weight, which shows its pattern but hides its scale.
    scale: each weight becomes a scale x scale block (nearest-neighbor, so the pixels stay crisp).
    gap: white pixels between tiles.
    """
    filters = weights[:, 0]  # [F, K, K]
    if vmax is None:
        limit = np.abs(filters).max(axis=(1, 2), keepdims=True)  # [F, 1, 1]
        # An all-zero filter would divide by zero. Any positive limit renders it as the zero color.
        limit = np.maximum(limit, 1e-12)
    else:
        limit = vmax
    # [-limit, limit] -> [0, 1], with zero at the middle of the colormap
    unit = 0.5 + 0.5 * np.clip(filters / limit, -1.0, 1.0)
    rgba = np.asarray(colormaps["RdBu_r"](unit))  # [F, K, K, 4], floats in [0, 1]
    rgb = (rgba[..., :3] * 255).round().astype(np.uint8)  # [F, K, K, 3]
    rgb = rgb.repeat(scale, axis=1).repeat(scale, axis=2)  # [F, K*scale, K*scale, 3]

    spacer = np.full((rgb.shape[1], gap, 3), 255, dtype=np.uint8)
    tiles: list[np.ndarray] = []
    for index, tile in enumerate(rgb):
        if index > 0:
            tiles.append(spacer)
        tiles.append(tile)
    return np.concatenate(tiles, axis=1)
