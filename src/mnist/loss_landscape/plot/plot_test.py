import numpy as np

from mnist.loss_landscape.plot.plot import format_tick, round_log_ticks


def test_round_log_ticks_name_round_values_in_a_wide_range() -> None:
    assert round_log_ticks(0.014, 2.3) == [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def test_round_log_ticks_fall_back_to_spread_values_in_a_narrow_range() -> None:
    # The first training steps barely move the loss, so no round value fits inside
    ticks = round_log_ticks(2.2144, 2.3025)
    assert len(ticks) == 4
    assert ticks[0] == 2.2144 and np.isclose(ticks[-1], 2.3025)
    assert ticks == sorted(ticks)


def test_format_tick_keeps_at_most_four_significant_digits() -> None:
    assert format_tick(2.214403) == "2.214"
    assert format_tick(0.1) == "0.1"
    assert format_tick(2.0) == "2"
