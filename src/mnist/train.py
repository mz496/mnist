from pathlib import Path
import wandb

from mnist.config import load_default_config
from mnist.data import get_dataloaders
from mnist.evaluate import evaluate
from mnist.utils import set_seed
from mnist.model import LeNet5
import torch
from datetime import datetime

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
    name = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = wandb.init(project="mnist", name=name, config=config.model_dump())

    data = get_dataloaders(config.data)

    epoch_count = 10
    for epoch in range(epoch_count):
        batch_count = 0
        for batch in data.train:
            batch_images, batch_labels = batch

            images = batch_images.to(device)
            labels = batch_labels.to(device)

            # Compute loss
            logits = model(images)
            train_loss = loss_fn(logits, labels)

            # Zero the gradients for this step
            optimizer.zero_grad()

            # Backprop the gradients through the computation graph
            train_loss.backward()

            # Update weights using the computed gradients
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]

            # Log training loss every Nth batch
            batch_count += 1
            if batch_count % 10 == 0:
                wandb.log({"train/loss": train_loss})


    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{run.id}.pt"
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    main()
