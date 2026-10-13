pytest:
    uv run pytest

train:
    uv run python -m mnist.train

test checkpoint:
    uv run python -m mnist.test "{{checkpoint}}"