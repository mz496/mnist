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
# with_format(torch) makes the access return Tensors.
ds: DatasetDict = load_dataset("ylecun/mnist").with_format("torch")  # pyright: ignore[reportUnknownMemberType] Bad stubs from the datasets lib.
class InputDatasetItem(TypedDict):
    image: Tensor # [1, 28, 28], of uint8s (0-255)
    label: Tensor # 0-dim

@dataclass
class Dataloaders:
    train: DataLoader[InputDatasetItem]
    val: DataLoader[InputDatasetItem]
    test: DataLoader[InputDatasetItem]

# Mapping from input data. 
def collate(inputs: list[InputDatasetItem]) -> tuple[Tensor, Tensor]:
    images = torch.stack([i["image"] for i in inputs]).float() / 255  # [B, 1, 28, 28], of float32s
    labels = torch.stack([i["label"] for i in inputs])  # [B], int64
    return images, labels
    
def get_dataloaders(data_config: DataConfig) -> Dataloaders:
    split = ds["train"].train_test_split(test_size=data_config.validation_set_size, seed=data_config.validation_set_split_seed)
    train_arrow_dataset: ArrowDataset = split["train"]
    val_arrow_dataset: ArrowDataset = split["test"]
    test_arrow_dataset: ArrowDataset = ds["test"]

    # Treat the HF Arrow-backed datasets directly as Torch's dataset type
    train_loader = DataLoader(cast(TorchDataset[InputDatasetItem], train_arrow_dataset), batch_size=data_config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(cast(TorchDataset[InputDatasetItem], val_arrow_dataset), batch_size=data_config.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(cast(TorchDataset[InputDatasetItem], test_arrow_dataset), batch_size=data_config.batch_size, shuffle=False, collate_fn=collate)

    return Dataloaders(train=train_loader, val=val_loader, test=test_loader)

