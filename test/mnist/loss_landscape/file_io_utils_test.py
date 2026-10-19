from pathlib import Path

from mnist.loss_landscape.file_io_utils import format_tags, viz_animations_dir, viz_filename, viz_plots_dir


def test_format_tags_names_match_cli_flags_and_skip_unset_params() -> None:
    assert format_tags(mode="2d", trajectory_plot=True, max_step=600) == "mode=2d_trajectory-plot=true_max-step=600"
    assert format_tags(mode="1d", trajectory_plot=False, max_step=None, seed=0) == "mode=1d_seed=0"


def test_viz_dirs_are_namespaced_by_run_id() -> None:
    assert viz_plots_dir("run") == Path("viz") / "run" / "plots"
    assert viz_animations_dir("run") == Path("viz") / "run" / "animations"


def test_viz_filename_starts_with_the_stamp_and_skips_empty_tags() -> None:
    assert viz_filename("20261016-101500", "mode=2d", "value=loss", ext="png") == "20261016-101500_mode=2d_value=loss.png"
    assert viz_filename("20261016-101500", "mode=1d_seed=0", "", ext="png") == "20261016-101500_mode=1d_seed=0.png"
