import wandb

from mnist.config import load_default_config
from mnist.data import get_dataloaders
from mnist.utils import set_seed
from mnist.model import LeNet5
import torch

def main():
    config = load_default_config()
    print("USING MODEL CONFIG")
    print(config)

    set_seed(config.train.seed)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("MOVING MODEL TO DEVICE:", device)
    model = LeNet5().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=config.train.learning_rate, momentum=config.train.momentum)
    loss_fn = torch.nn.CrossEntropyLoss()
    wandb.init(project="mnist", config=config.model_dump())

    data = get_dataloaders(config.data)

    batch_count = 0
    for batch in data.train:
        batch_images, batch_labels = batch

        images = batch_images.to(device)
        labels = batch_labels.to(device)

        # Compute loss
        logits = model(images)
        loss = loss_fn(logits, labels)

        # Zero the gradients for this step
        optimizer.zero_grad()

        # Backprop the gradients through the computation graph
        loss.backward()

        # Update weights using the computed gradients
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]

        # Log
        batch_count += 1
        if batch_count % 10 == 0:
            wandb.log({"train/loss": loss})


if __name__ == "__main__":
    main()
