
from typing import cast
from mnist.config import load_default_config
from mnist.data import get_dataloaders
from mnist.evaluate import evaluate
from mnist.model import LeNet5
from mnist.utils import get_device
import argparse
import torch
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test set")
    parser.add_argument("checkpoint", type=Path, help="path to a .pt state_dict, e.g. checkpoints/abc123.pt")
    return parser.parse_args()

def load_checkpoint(path: Path, device: str) -> LeNet5:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    model = LeNet5().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    device = get_device()
    print("MOVING MODEL TO DEVICE:", device)
    model = load_checkpoint(checkpoint_path, device)

    config = load_default_config()
    print("USING MODEL CONFIG")
    print(config)

    data = get_dataloaders(config.data)

    # Load the checkpoint and eval on it
    test_loss, test_accuracy = evaluate(model, data.test, device)
    print(f"TEST LOSS: {test_loss}")
    print(f"TEST ACC:  {test_accuracy}")


if __name__ == "__main__":
    main()