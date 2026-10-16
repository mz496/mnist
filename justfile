pytest:
    uv run pytest

train:
    uv run python -m mnist.train

test checkpoint:
    uv run python -m mnist.test "{{checkpoint}}"

loss-landscape-plot run_id *flags:
    uv run python -m mnist.loss_landscape.plot "{{run_id}}" {{flags}}

loss-landscape-animate run_id *flags:
    uv run python -m mnist.loss_landscape.animate "{{run_id}}" {{flags}}
