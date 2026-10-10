from typing import TypedDict, cast
from PIL.Image import Image
from datasets import Dataset as ArrowDataset, DatasetDict, load_dataset
from pydantic import BaseModel
from torch.utils.data import TensorDataset, DataLoader, Dataset as TorchDataset
from torch import Tensor
import torch
from dataclasses import dataclass

from torchvision import transforms
from mnist.config import DataConfig

# Encoding the structure of https://huggingface.co/datasets/ylecun/mnist
ds: DatasetDict = load_dataset("ylecun/mnist").with_format("torch")  # pyright: ignore[reportUnknownMemberType] Bad stubs from the datasets lib.

class Batch(TypedDict):
    image: Tensor # [B, 1, 28, 28], of uint8s (0-255)
    label: Tensor # [B]

@dataclass
class Dataloaders:
    train: DataLoader[Batch]
    val: DataLoader[Batch]
    test: DataLoader[Batch]

def get_dataloaders(data_config: DataConfig) -> Dataloaders:
    split = ds["train"].train_test_split(test_size=data_config.validation_set_size, seed=data_config.validation_set_split_seed)
    train: ArrowDataset = split["train"]
    val: ArrowDataset = split["test"]
    test: ArrowDataset = ds["test"]

    train_loader = DataLoader(cast(TorchDataset[Batch], train), batch_size=data_config.batch_size, shuffle=True)
    val_loader = DataLoader(cast(TorchDataset[Batch], val), batch_size=data_config.batch_size, shuffle=False)
    test_loader = DataLoader(cast(TorchDataset[Batch], test), batch_size=data_config.batch_size, shuffle=False)

    return Dataloaders(train=train_loader, val=val_loader, test=test_loader)

