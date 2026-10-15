pytest:
    uv run pytest

train:
    uv run python -m mnist.train

test checkpoint:
    uv run python -m mnist.test "{{checkpoint}}"

loss-landscape-compute run_id *flags:
    uv run python -m mnist.loss_landscape.compute_surface "{{run_id}}" {{flags}}

loss-landscape-plot npz *flags:
    uv run python -m mnist.loss_landscape.plot "{{npz}}" {{flags}}
