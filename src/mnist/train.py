import wandb
from .model import LeNet5
import torch

def main():
    model = LeNet5()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    """
    run = wandb.init(project="mnist-smoke-test")
    wandb.log({"train/loss": 0})
    run.summary["final_val_acc"] = 1
    wandb.finish()
    print("Hello from mnist!")
    """


if __name__ == "__main__":
    main()
