import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mnist.scripts import make_layer_view_video as video


def _png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _make_run(root: Path, steps_left: list[int], steps_right: list[int], left_size=(40, 30), right_size=(60, 20)) -> Path:
    run = root / "run-20260101_000000-abc"
    images = run / "files" / "media" / "images"
    for step in steps_left:
        _png(images / "layer_view_conv1" / f"relu_{step}_0123456789abcdef0123.png", left_size, (255, 0, 0))
    for step in steps_right:
        _png(images / "layer_view_conv2" / f"relu_{step}_fedcba9876543210fedc.png", right_size, (0, 0, 255))
    # Frames of the other view must be left alone
    _png(images / "layer_view_conv1" / "preact_5_0123456789abcdef0123.png", left_size, (0, 255, 0))
    return run


def test_find_frames_reads_the_step_from_the_file_name_and_ignores_other_views(tmp_path):
    run = _make_run(tmp_path, [0, 2, 10], [0])
    frames = video.find_frames(run / "files/media/images/layer_view_conv1", "relu")
    assert sorted(frames) == [0, 2, 10]
    assert frames[10].name.startswith("relu_10_")
    preact = video.find_frames(run / "files/media/images/layer_view_conv1", "preact")
    assert sorted(preact) == [5]


def test_steps_sort_by_number_not_by_text():
    left = {step: Path(f"l{step}") for step in (2, 10, 1)}
    right = {step: Path(f"r{step}") for step in (10, 2, 1)}
    assert video.paired_steps(left, right) == [1, 2, 10]


def test_a_step_with_an_image_on_only_one_side_is_not_paired():
    left = {1: Path("a"), 2: Path("b"), 3: Path("c")}
    right = {1: Path("x"), 3: Path("z"), 4: Path("w")}
    assert video.paired_steps(left, right) == [1, 3]


def test_the_frame_size_fits_both_images_and_is_even():
    width, height = video.frame_size((41, 30), (59, 21))
    assert width >= 41 + video.GAP + 59 and height >= video.HEADER_HEIGHT + 30
    assert width % 2 == 0 and height % 2 == 0


def test_a_frame_has_the_title_and_the_step_on_two_lines_on_top_and_the_images_below_side_by_side():
    left = Image.new("RGB", (40, 30), (255, 0, 0))
    right = Image.new("RGB", (60, 20), (0, 0, 255))
    size = video.frame_size(left.size, right.size)
    frame = video.compose_frame(left, right, 7, size, "run-test")
    assert frame.shape == (size[1], size[0], 3)
    # Text on both lines of the header band
    assert (frame[: video.LINE_HEIGHT] != 255).any()
    assert (frame[video.LINE_HEIGHT : video.HEADER_HEIGHT] != 255).any()
    # The left image starts under the header at the left edge, the right image after the gap
    assert (frame[video.HEADER_HEIGHT + 5, 5] == (255, 0, 0)).all()
    assert (frame[video.HEADER_HEIGHT + 5, 40 + video.GAP + 5] == (0, 0, 255)).all()
    # The gap is white
    assert (frame[video.HEADER_HEIGHT + 5, 40 + 2] == 255).all()
    # A different step changes only the second line, and a different title changes only the first
    other_step = video.compose_frame(left, right, 8, size, "run-test")
    assert np.array_equal(frame[: video.LINE_HEIGHT], other_step[: video.LINE_HEIGHT])
    assert not np.array_equal(frame[video.LINE_HEIGHT : video.HEADER_HEIGHT], other_step[video.LINE_HEIGHT : video.HEADER_HEIGHT])
    other_title = video.compose_frame(left, right, 7, size, "run-other")
    assert not np.array_equal(frame[: video.LINE_HEIGHT], other_title[: video.LINE_HEIGHT])
    assert np.array_equal(frame[video.LINE_HEIGHT : video.HEADER_HEIGHT], other_title[video.LINE_HEIGHT : video.HEADER_HEIGHT])


def test_an_unknown_run_is_an_error_that_lists_the_newest_runs(tmp_path, capsys):
    _make_run(tmp_path, [0], [0])
    assert video.main(["run-nope", "--wandb-dir", str(tmp_path)]) == 1
    assert "run-20260101_000000-abc" in capsys.readouterr().err


def test_a_run_without_the_layer_view_images_is_an_error(tmp_path, capsys):
    run = tmp_path / "run-20260101_000000-abc"
    (run / "files" / "media" / "images").mkdir(parents=True)
    assert video.main([run.name, "--wandb-dir", str(tmp_path)]) == 1
    assert "layer_view_conv1" in capsys.readouterr().err


def test_no_matching_frames_on_both_sides_is_an_error(tmp_path, capsys):
    run = _make_run(tmp_path, [1, 3], [2, 4])
    assert video.main([run.name, "--wandb-dir", str(tmp_path)]) == 1
    assert "no relu_N images" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg and ffprobe")
def test_it_writes_an_mp4_named_after_the_run_with_one_frame_per_paired_step_into_outputs(tmp_path):
    run = _make_run(tmp_path, [0, 1, 2, 3], [0, 1, 2, 3, 9], left_size=(41, 30), right_size=(59, 21))
    assert video.main([run.name, "--wandb-dir", str(tmp_path), "--fps", "4"]) == 0
    out = run / "outputs" / f"{run.name}-out.mp4"
    assert out.is_file()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=codec_name,nb_read_frames,width,height", "-of", "default=nw=1", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout
    fields = dict(line.split("=") for line in probe.split())
    assert fields["codec_name"] == "h264"
    assert fields["nb_read_frames"] == "4"
    assert int(fields["width"]) % 2 == 0 and int(fields["height"]) % 2 == 0
    # Step 9 has no left image, so it is skipped and the frames are not written anywhere else
    assert sorted(path.name for path in (run / "outputs").iterdir()) == [f"{run.name}-out.mp4"]
