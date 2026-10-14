pytest:
    uv run pytest

train:
    uv run python -m mnist.train

test checkpoint:
    uv run python -m mnist.test "{{checkpoint}}"

loss-landscape-compute checkpoint:
    uv run python -m mnist.loss_landscape.compute_surface "{{checkpoint}}"

loss-landscape-plot npz:
    uv run python -m mnist.loss_landscape.plot "{{npz}}"
