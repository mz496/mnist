# Runs unit tests.
pytest:
    uv run pytest

# Trains the network with the default config. Leaves artifacts in `checkpoints/`.
train:
    uv run python -m mnist.train

# Runs training on different combinations of hyperparameters, treated as overrides of the config. 
# e.g. just hyperparameter-sweep train.seed=1,2,3,4 --workers 4 --name seed-check
hyperparameter-sweep *args:
    uv run python -m mnist.hyperparameter_sweep {{args}}

# Runs the given training checkpoint file against the test set.
test checkpoint:
    uv run python -m mnist.test "{{checkpoint}}"

# Creates a plot on the training run ID.
# Leaves visualization artifacts in `viz/` and additional data artifacts in `checkpoints/` as a kind of cache, so they can be reused if run on the same parameters.
loss-landscape-plot run_id *flags:
    uv run python -m mnist.loss_landscape.plot.plot "{{run_id}}" {{flags}}

# Creates an animation from multiple plots on the training run ID.
# Leaves visualization artifacts in `viz/` and additional data artifacts in `checkpoints/` as a kind of cache, so they can be reused if run on the same parameters.
loss-landscape-animate run_id *flags:
    uv run python -m mnist.loss_landscape.animate.animate "{{run_id}}" {{flags}}
