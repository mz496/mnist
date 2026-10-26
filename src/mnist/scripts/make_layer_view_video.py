"""
Makes an mp4 from the layer view images that W&B saved in a local run folder.

Each frame puts the conv1 image on the left and the conv2 image on the right, both for the same logged step. Across the
top, the first line is the run folder name and the second line is "step=N". The images are read from the run folder, so
nothing has to be downloaded or saved first:

    wandb/<run dir>/files/media/images/layer_view_conv1/relu_N_<hash>.png
    wandb/<run dir>/files/media/images/layer_view_conv2/relu_N_<hash>.png

and the video is written to wandb/<run dir>/outputs/<run dir>-out.mp4.

N is the number in the file name. W&B counts every wandb.log call, including the validation logs, so N is the index of
the log call. The frames are in the order the steps were logged either way.

Usage, from the repo root:
    uv run python -m mnist.scripts.make_layer_view_video run-20261026_125406-1rchthqz
    uv run python -m mnist.scripts.make_layer_view_video run-20261026_125406-1rchthqz --fps 10 --view preact
or through the justfile: just layer-view-video run-20261026_125406-1rchthqz

Needs ffmpeg on the PATH.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
LEFT_KEY = "layer_view_conv1"
RIGHT_KEY = "layer_view_conv2"
# White pixels between the two images, and the height of the band that holds the two lines of header text.
GAP = 24
LINE_HEIGHT = 48
HEADER_HEIGHT = 2 * LINE_HEIGHT
FONT_SIZE = 30

def find_frames(images_dir: Path, view: str) -> dict[int, Path]:
    """Maps N to the file for every `{view}_N_<hash>.png` in `images_dir`. Other files are ignored."""
    pattern = re.compile(rf"^{re.escape(view)}_(\d+)_[0-9a-f]+\.png$")
    frames: dict[int, Path] = {}
    for path in images_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            frames[int(match.group(1))] = path
    return frames

def paired_steps(left: dict[int, Path], right: dict[int, Path]) -> list[int]:
    """The step numbers present on both sides, in order. A step with only one image has nothing to sync with, so it is skipped."""
    return sorted(left.keys() & right.keys())

def compose_frame(left: Image.Image, right: Image.Image, step: int, size: tuple[int, int], title: str) -> np.ndarray:
    """
    One video frame as uint8 RGB [height, width, 3] of the given (width, height): across the top the title, then
    "step=N" on a second line, then below that the left image, a gap and the right image, with white padding.
    The size must be even and at least as large as the content.
    """
    width = size[0]
    canvas = Image.new("RGB", size, (255, 255, 255))
    canvas.paste(left, (0, HEADER_HEIGHT))
    canvas.paste(right, (left.width + GAP, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=FONT_SIZE)
    for line_index, text in enumerate((title, f"step={step}")):
        left_edge, top, right_edge, bottom = draw.textbbox((0, 0), text, font=font)
        x = (width - (right_edge - left_edge)) // 2 - left_edge
        y = line_index * LINE_HEIGHT + (LINE_HEIGHT - (bottom - top)) // 2 - top
        draw.text((x, y), text, fill=(40, 40, 40), font=font)
    return np.asarray(canvas, dtype=np.uint8)

def frame_size(left_size: tuple[int, int], right_size: tuple[int, int]) -> tuple[int, int]:
    """The (width, height) that fits both images and the header, rounded up to even numbers, which H.264 requires."""
    width = left_size[0] + GAP + right_size[0]
    height = HEADER_HEIGHT + max(left_size[1], right_size[1])
    return width + width % 2, height + height % 2

def write_video(frames: list[tuple[int, Path, Path]], output: Path, fps: int, title: str) -> None:
    """Pipes composed frames straight into ffmpeg, so no frame images are written to disk. `title` is the first header line."""
    with Image.open(frames[0][1]) as first_left, Image.open(frames[0][2]) as first_right:
        size = frame_size(first_left.size, first_right.size)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for index, (step, left_path, right_path) in enumerate(frames, start=1):
            with Image.open(left_path) as left, Image.open(right_path) as right:
                if frame_size(left.size, right.size) != size:
                    raise ValueError(f"step {step}: the images are a different size from the first frame, so the video cannot be built")
                process.stdin.write(compose_frame(left.convert("RGB"), right.convert("RGB"), step, size, title).tobytes())
            if index % 25 == 0 or index == len(frames):
                print(f"  {index}/{len(frames)} frames", flush=True)
    finally:
        process.stdin.close()
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {returncode}")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Make an mp4 of the conv1 and conv2 layer view images of a W&B run.")
    parser.add_argument("run_dir", help="the run folder name inside wandb/, such as run-20261026_125406-1rchthqz")
    parser.add_argument("--view", default="relu", help="which image to use from each layer, the prefix of the file names (default: relu)")
    parser.add_argument("--fps", type=int, default=5, help="frames per second (default: 5)")
    parser.add_argument("--wandb-dir", type=Path, default=REPO_ROOT / "wandb", help="where the run folders are (default: wandb/ in the repo)")
    args = parser.parse_args(argv)

    run = args.wandb_dir / args.run_dir
    if not run.is_dir():
        known = sorted(path.name for path in args.wandb_dir.glob("run-*"))[-5:] if args.wandb_dir.is_dir() else []
        print(f"error: there is no run folder {run}. The newest ones are: {', '.join(known) or 'none'}", file=sys.stderr)
        return 1
    images = run / "files" / "media" / "images"
    for key in (LEFT_KEY, RIGHT_KEY):
        if not (images / key).is_dir():
            print(f"error: {images / key} does not exist. Was {key} logged in this run?", file=sys.stderr)
            return 1
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg is not on the PATH", file=sys.stderr)
        return 1

    left = find_frames(images / LEFT_KEY, args.view)
    right = find_frames(images / RIGHT_KEY, args.view)
    steps = paired_steps(left, right)
    if not steps:
        print(f"error: no {args.view}_N images are present in both {LEFT_KEY} and {RIGHT_KEY}", file=sys.stderr)
        return 1
    skipped = len(left.keys() ^ right.keys())
    if skipped:
        print(f"note: skipped {skipped} step(s) that have an image on only one side")

    output = run / "outputs" / f"{run.name}-out.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"making {output} from {len(steps)} frames, steps {steps[0]} to {steps[-1]}")
    write_video([(step, left[step], right[step]) for step in steps], output, args.fps, run.name)
    print(f"wrote {output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
