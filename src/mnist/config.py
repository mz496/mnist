from pathlib import Path
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

class DataConfig(BaseModel):
    batch_size: int
    validation_set_size: int
    validation_set_split_seed: int


class TrainConfig(BaseModel):
    seed: int

class Config(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra='forbid',
    )

    # TODO fiddle with this once training loop is complete.
    # model: ModelConfig
    data: DataConfig
    train: TrainConfig

def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e

    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} must contain a YAML mapping at the top level, got {type(raw).__name__}"
        )

    try:
        return Config(**raw)
    except ValidationError as e:
        raise ValueError(f"Invalid config in {path}:\n{e}") from e


def load_default_config() -> Config:
    return load_config("configs/default.yaml")
