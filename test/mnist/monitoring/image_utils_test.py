import numpy as np

import pytest

from mnist.monitoring.image_utils import (
    GUTTER_WIDTH,
    LABEL_HEIGHT,
    ColumnGroup,
    add_axis_labels,
    center_in_width,
    join_row,
    stack_rows,
    text_cell,
    tile_row,
    to_uint8_rgb,
)


def test_to_uint8_rgb_drops_alpha_and_scales_to_bytes():
    rgba = np.array([[[0.0, 0.5, 1.0, 0.3]]])
    rgb = to_uint8_rgb(rgba)
    assert rgb.dtype == np.uint8
    assert rgb.tolist() == [[[0, 128, 255]]]


def test_tile_row_upscales_each_tile_and_puts_a_white_gap_between_them():
    tiles = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    row = tile_row(tiles, scale=3, gap=2)
    assert row.shape == (6, 3 * 6 + 2 * 2, 3)
    assert (row[:, 6:8] == 255).all()


def test_stack_rows_puts_a_white_gap_between_rows():
    rows = [np.zeros((4, 5, 3), dtype=np.uint8), np.zeros((4, 5, 3), dtype=np.uint8)]
    stacked = stack_rows(rows, gap=3)
    assert stacked.shape == (4 + 3 + 4, 5, 3)
    assert (stacked[4:7] == 255).all()


def test_join_row_puts_a_white_gap_between_parts():
    left = np.zeros((4, 3, 3), dtype=np.uint8)
    right = np.zeros((4, 5, 3), dtype=np.uint8)
    joined = join_row([left, right], gap=2)
    assert joined.shape == (4, 3 + 2 + 5, 3)
    assert (joined[:, 3:5] == 255).all()


def test_join_row_centers_a_shorter_part_on_white():
    tall = np.zeros((6, 2, 3), dtype=np.uint8)
    short = np.zeros((2, 2, 3), dtype=np.uint8)
    joined = join_row([tall, short], gap=1)
    assert joined.shape == (6, 2 + 1 + 2, 3)
    short_side = joined[:, 3:]
    # Rows 2 and 3 hold the short part, the rest is white padding
    assert (short_side[2:4] == 0).all()
    assert (short_side[:2] == 255).all() and (short_side[4:] == 255).all()


def test_text_cell_is_a_white_cell_of_the_requested_size_with_text_in_it():
    cell = text_cell("12", width=30, height=LABEL_HEIGHT)
    assert cell.shape == (LABEL_HEIGHT, 30, 3)
    assert cell.dtype == np.uint8
    assert (cell < 255).any()
    # The text is centered, so the outer columns stay white
    assert (cell[:, :4] == 255).all() and (cell[:, -4:] == 255).all()


def test_text_cell_with_no_text_is_all_white():
    assert (text_cell("", width=10, height=10) == 255).all()


def _labeled_body(columns: int, rows: int, cell_width: int, row_height: int, gap: int) -> np.ndarray:
    return np.zeros((rows * row_height + (rows - 1) * gap, columns * cell_width + (columns - 1) * gap, 3), dtype=np.uint8)


def test_add_axis_labels_puts_two_header_lines_above_and_a_label_column_to_the_left():
    body = _labeled_body(columns=4, rows=2, cell_width=20, row_height=15, gap=2)
    groups = [ColumnGroup("out", ["0", "1", "2", "3"], 20)]
    image = add_axis_labels(body, groups, ["a", "b"], "in", row_height=15, gap=2, section_gap=8)
    assert image.shape == (body.shape[0] + 2 * LABEL_HEIGHT, body.shape[1] + GUTTER_WIDTH, 3)
    assert (image[2 * LABEL_HEIGHT:, GUTTER_WIDTH:] == body).all()


def test_add_axis_labels_joins_column_groups_with_the_section_gap():
    # Two groups of different tile widths, as in a row with an input side and a maps side
    body = np.zeros((30, (2 * 20 + 2) + 8 + (3 * 10 + 2 * 2), 3), dtype=np.uint8)
    groups = [ColumnGroup("in", ["0", "1"], 20), ColumnGroup("out", ["0", "1", "2"], 10)]
    image = add_axis_labels(body, groups, ["x"], "c", row_height=30, gap=2, section_gap=8)
    assert image.shape[1] == body.shape[1] + GUTTER_WIDTH


def test_add_axis_labels_rejects_a_body_that_does_not_match_the_labels():
    body = _labeled_body(columns=4, rows=2, cell_width=20, row_height=15, gap=2)
    with pytest.raises(ValueError):
        add_axis_labels(body, [ColumnGroup("out", ["0", "1", "2"], 20)], ["a", "b"], "in", row_height=15, gap=2, section_gap=8)
    with pytest.raises(ValueError):
        add_axis_labels(body, [ColumnGroup("out", ["0", "1", "2", "3"], 20)], ["a"], "in", row_height=15, gap=2, section_gap=8)


def test_center_in_width_pads_both_sides_with_white():
    tiles = np.zeros((2, 3, 4, 3), dtype=np.uint8)
    padded = center_in_width(tiles, 10)
    assert padded.shape == (2, 3, 10, 3)
    assert (padded[:, :, 3:7] == 0).all()
    assert (padded[:, :, :3] == 255).all() and (padded[:, :, 7:] == 255).all()


def test_center_in_width_refuses_a_cell_narrower_than_the_tile():
    with pytest.raises(ValueError):
        center_in_width(np.zeros((1, 3, 4, 3), dtype=np.uint8), 3)


def test_tile_row_can_give_every_tile_a_wider_cell():
    tiles = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    row = tile_row(tiles, scale=2, gap=1, cell_width=8)
    assert row.shape == (4, 3 * 8 + 2 * 1, 3)
    # Each 4 pixel wide tile is centered in its 8 pixel cell
    assert (row[:, 2:6] == 0).all()
    assert (row[:, :2] == 255).all() and (row[:, 6:9] == 255).all()
