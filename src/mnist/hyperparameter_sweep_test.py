import pytest

from mnist.hyperparameter_sweep import flatten, parse_axes, tag_for, to_sweep_config, validate_axes


def test_parse_axes_parses_yaml_types():
    assert parse_axes(["train.seed=1,2", "train.learning_rate=0.01"]) == {
        "train.seed": [1, 2],
        "train.learning_rate": [0.01],
    }


def test_to_sweep_config_nests_parameters():
    config = to_sweep_config({"train.learning_rate": [0.001, 0.01], "train.seed": [1, 2, 3]})
    assert config["method"] == "grid"
    assert config["parameters"] == {
        "train": {
            "parameters": {
                "learning_rate": {"values": [0.001, 0.01]},
                "seed": {"values": [1, 2, 3]},
            }
        }
    }


def test_flatten_round_trips_nested_sweep_values():
    assert flatten({"train": {"learning_rate": 0.01, "seed": 2}}) == {
        "train.learning_rate": 0.01,
        "train.seed": 2,
    }


def test_tag_for_drops_section():
    assert tag_for({"train.learning_rate": 0.01, "train.seed": 2}) == "learning_rate=0.01,seed=2"


def test_validate_axes_rejects_unknown_key():
    with pytest.raises(KeyError):
        validate_axes({"train.learning_rte": [0.1]})
