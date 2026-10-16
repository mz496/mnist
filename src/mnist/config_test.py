import pytest
from pydantic import ValidationError

from mnist.config import Config, DataConfig, TrainConfig, with_overrides


def make_config() -> Config:
    return Config(
        data=DataConfig(batch_size=64, validation_set_size=5000, validation_set_split_seed=42),
        train=TrainConfig(seed=1, learning_rate=0.01, momentum=0.9, epoch_count=10),
    )


def test_with_overrides_replaces_value_and_leaves_original():
    config = make_config()
    new = with_overrides(config, {"train.learning_rate": 0.1, "train.seed": 5})
    assert new.train.learning_rate == 0.1
    assert new.train.seed == 5
    assert config.train.learning_rate == 0.01


def test_with_overrides_rejects_unknown_key():
    with pytest.raises(KeyError):
        with_overrides(make_config(), {"train.learning_rte": 0.1})


def test_with_overrides_rejects_wrong_type():
    with pytest.raises(ValidationError):
        with_overrides(make_config(), {"train.momentum": "fast"})
