from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def to_uint8_rgb(rgba: np.ndarray) -> np.ndarray:
    """Drops the alpha channel from colormap output (floats in [0, 1]) and converts to uint8 RGB."""
    return (np.asarray(rgba)[..., :3] * 255).round().astype(np.uint8)

def tile_row(tiles: np.ndarray, scale: int, gap: int, cell_width: int | None = None) -> np.ndarray:
    """
    Lays [T, h, w, 3] uint8 tiles side by side in one row, each upscaled to (h * scale) x (w * scale)
    by nearest-neighbor, with `gap` white pixels between tiles.

    cell_width: give every tile a cell this many pixels wide, with the tile centered in it on white.
                None uses the tile's own width. Must be at least w * scale.
    Returns uint8 RGB [h * scale, T * cell_width + (T - 1) * gap, 3].
    """
    tiles = tiles.repeat(scale, axis=1).repeat(scale, axis=2)
    if cell_width is not None:
        tiles = center_in_width(tiles, cell_width)
    spacer = np.full((tiles.shape[1], gap, 3), 255, dtype=np.uint8)
    row: list[np.ndarray] = []
    for index, tile in enumerate(tiles):
        if index > 0:
            row.append(spacer)
        row.append(tile)
    return np.concatenate(row, axis=1)

def center_in_width(tiles: np.ndarray, width: int) -> np.ndarray:
    """Pads [T, h, w, 3] uint8 tiles with white on the left and right to [T, h, width, 3], centering each one."""
    tile_width = tiles.shape[2]
    if width < tile_width:
        raise ValueError(f"cannot fit tiles {tile_width} pixels wide in a cell {width} pixels wide")
    left = (width - tile_width) // 2
    padded = np.full((tiles.shape[0], tiles.shape[1], width, 3), 255, dtype=np.uint8)
    padded[:, :, left:left + tile_width] = tiles
    return padded

def stack_rows(rows: list[np.ndarray], gap: int) -> np.ndarray:
    """Stacks uint8 RGB images of the same width top to bottom, with `gap` white pixels between them."""
    spacer = np.full((gap, rows[0].shape[1], 3), 255, dtype=np.uint8)
    stacked: list[np.ndarray] = []
    for index, row in enumerate(rows):
        if index > 0:
            stacked.append(spacer)
        stacked.append(row)
    return np.concatenate(stacked, axis=0)

def join_row(parts: list[np.ndarray], gap: int) -> np.ndarray:
    """
    Joins uint8 RGB images left to right with `gap` white pixels between them.
    Parts can differ in height. A shorter part is centered vertically on white.
    """
    height = max(part.shape[0] for part in parts)
    padded: list[np.ndarray] = []
    for index, part in enumerate(parts):
        if index > 0:
            padded.append(np.full((height, gap, 3), 255, dtype=np.uint8))
        top = (height - part.shape[0]) // 2
        canvas = np.full((height, part.shape[1], 3), 255, dtype=np.uint8)
        canvas[top:top + part.shape[0]] = part
        padded.append(canvas)
    return np.concatenate(padded, axis=1)

# Height in pixels of one line of label text, and width of the column of row labels on the left.
LABEL_HEIGHT = 13
GUTTER_WIDTH = 40
_FONT = ImageFont.load_default()
_TEXT_COLOR = (60, 60, 60)

def text_cell(text: str, width: int, height: int = LABEL_HEIGHT) -> np.ndarray:
    """A white uint8 RGB cell of the given size with `text` centered in it. Text wider than the cell is cut off at its edges."""
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=_FONT)
    x = (width - (right - left)) // 2 - left
    y = (height - (bottom - top)) // 2 - top
    draw.text((x, y), text, fill=_TEXT_COLOR, font=_FONT)
    return np.asarray(canvas, dtype=np.uint8)

class ColumnGroup(NamedTuple):
    """
    A run of equal-width columns that share a title, for add_axis_labels.
    - title: text centered over the whole group.
    - labels: one text per column, centered over that column.
    - cell_width: pixels per column. Columns are separated by the same gap as in the body.
    """
    title: str
    labels: list[str]
    cell_width: int

def add_axis_labels(
    body: np.ndarray,
    groups: Sequence[ColumnGroup],
    row_labels: Sequence[str],
    corner: str,
    row_height: int,
    gap: int,
    section_gap: int,
) -> np.ndarray:
    """
    Frames `body`, an image made of rows and column groups, with text.
    - Above: a title for each column group, then a label for each column.
    - Left: a label for each row, with `corner` text in the corner above them.
    `body` must be laid out as the groups and rows say: the groups joined left to right by `section_gap`, with `gap`
    between the columns of a group, and the rows stacked top to bottom with `gap` between them, each `row_height` tall.
    Raises ValueError if the body does not match.
    """
    titles = join_row([text_cell(group.title, _group_width(group, gap)) for group in groups], section_gap)
    columns = join_row(
        [join_row([text_cell(label, group.cell_width) for label in group.labels], gap) for group in groups],
        section_gap,
    )
    header = stack_rows([titles, columns], 0)
    rows = stack_rows([text_cell(label, GUTTER_WIDTH, row_height) for label in row_labels], gap)
    if header.shape[1] != body.shape[1] or rows.shape[0] != body.shape[0]:
        raise ValueError(
            f"labels span {(rows.shape[0], header.shape[1])} pixels (rows, columns) but the body is {body.shape[:2]}"
        )
    left = stack_rows([text_cell(corner, GUTTER_WIDTH, header.shape[0]), rows], 0)
    right = stack_rows([header, body], 0)
    return join_row([left, right], 0)

def _group_width(group: ColumnGroup, gap: int) -> int:
    count = len(group.labels)
    return count * group.cell_width + (count - 1) * gap
