import argparse
import multiprocessing
from typing import Any

import wandb
import yaml

from mnist.config import load_default_config, with_overrides
from mnist.train import train
from mnist.utils import timestamp

PROJECT = "mnist"


def parse_axes(args: list[str]) -> dict[str, list[Any]]:
    """
    ["train.seed=1,2,3", "train.learning_rate=0.01"] -> {"train.seed": [1, 2, 3], "train.learning_rate": [0.01]}
    Values are parsed as YAML, so 0.01 is a float and 1 is an int.
    """
    axes: dict[str, list[Any]] = {}
    for arg in args:
        key, sep, values = arg.partition("=")
        if not sep or not values:
            raise ValueError(f"expected key=v1,v2,..., got {arg!r}")
        axes[key] = [yaml.safe_load(v) for v in values.split(",")]
    return axes


def validate_axes(axes: dict[str, list[Any]]) -> None:
    """
    Raises if any key is unknown or any value fails config validation, before a sweep is created.
    """
    base = load_default_config()
    for key, values in axes.items():
        for value in values:
            with_overrides(base, {key: value})


def to_sweep_config(axes: dict[str, list[Any]], name: str | None = None) -> dict[str, Any]:
    """
    Builds a W&B grid sweep config. Dotted keys become nested parameters so that wandb.config
    mirrors our Config: {"train.seed": [1, 2]} -> parameters: {train: {parameters: {seed: {values: [1, 2]}}}}
    """
    parameters: dict[str, Any] = {}
    for dotted, values in axes.items():
        section, _, key = dotted.partition(".")
        parameters.setdefault(section, {"parameters": {}})["parameters"][key] = {"values": values}
    return {
        "name": f"sweep_{timestamp()}" + (f"_{name}" if name else ""),
        "method": "grid",
        "metric": {"name": "validation/loss", "goal": "minimize"},
        "parameters": parameters,
    }


def flatten(nested: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """
    {"train": {"seed": 2}} -> {"train.seed": 2}
    """
    flat: dict[str, Any] = {}
    for key, value in nested.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{dotted}."))  # pyright: ignore[reportUnknownArgumentType]
        else:
            flat[dotted] = value
    return flat


def tag_for(overrides: dict[str, Any]) -> str:
    """
    {"train.learning_rate": 0.01, "train.seed": 2} -> "learning_rate=0.01,seed=2"
    """
    return ",".join(f"{key.partition('.')[2]}={value}" for key, value in overrides.items())


def run_trial() -> None:
    """
    The function a sweep agent calls once per config: reads the values the sweep chose, applies
    them to the default config, and trains.
    """
    run = wandb.init()
    overrides = flatten(run.config.as_dict())
    config = with_overrides(load_default_config(), overrides)
    run.name = tag_for(overrides)
    # Also record the non-swept values so every run logs the full config.
    run.config.update(config.model_dump(), allow_val_change=True)
    # Sweeps compare many runs' curves; skip the large per-run snapshot files.
    train(config, run=run, save_snapshots=False)


def create_sweep(axes: dict[str, list[Any]], name: str | None = None) -> str:
    validate_axes(axes)
    sweep_config = to_sweep_config(axes, name)
    sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    print(f"created sweep {sweep_config['name']} ({sweep_id})")
    return sweep_id


def agent_worker(sweep_id: str, count: int | None) -> None:
    wandb.agent(sweep_id, function=run_trial, project=PROJECT, count=count)


def run_agents(sweep_id: str, workers: int, count: int | None) -> None:
    """
    Runs `workers` agents against the sweep. Each pulls configs until the grid is exhausted
    (or `count` runs), so workers split the grid between them.
    """
    if workers == 1:
        agent_worker(sweep_id, count)
        return
    # Separate processes (spawned, not forked) so each has its own W&B state.
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=agent_worker, args=(sweep_id, count)) for _ in range(workers)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    exit_codes = [process.exitcode for process in processes]
    if any(exit_codes):
        raise SystemExit(f"agent exit codes: {exit_codes}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep (Cartesian grid) via a W&B sweep.")
    parser.add_argument("axes", nargs="+", help="key=v1,v2,... e.g. train.seed=1,2,3")
    parser.add_argument("--name", default=None, help="appended to the sweep name")
    parser.add_argument("--workers", type=int, default=1, help="parallel agent processes")
    parser.add_argument("--count", type=int, default=None, help="max runs per agent")
    args = parser.parse_args()
    run_agents(create_sweep(parse_axes(args.axes), args.name), args.workers, args.count)


if __name__ == "__main__":
    main()
