pytest:
    uv run pytest

train:
    uv run python -m mnist.train

# e.g. just hyperparameter-sweep train.seed=1,2,3,4 --workers 4
hyperparameter-sweep *args:
    uv run python -m mnist.hyperparameter_sweep {{args}}

test checkpoint:
    uv run python -m mnist.test "{{checkpoint}}"

loss-landscape-plot run_id *flags:
    uv run python -m mnist.loss_landscape.plot.plot "{{run_id}}" {{flags}}

loss-landscape-animate run_id *flags:
    uv run python -m mnist.loss_landscape.animate.animate "{{run_id}}" {{flags}}
