pytest:
    uv run pytest

train:
    uv run python -m mnist.train

test checkpoint:
    uv run python -m mnist.test "{{checkpoint}}"

loss-landscape-compute checkpoint mode="1d":
    uv run python -m mnist.loss_landscape.compute_surface "{{checkpoint}}" --mode {{mode}}

loss-landscape-plot npz style="contour":
    uv run python -m mnist.loss_landscape.plot "{{npz}}" --style {{style}}
