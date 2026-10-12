from pathlib import Path
from typing import Callable, cast
import wandb

from mnist.config import load_default_config
from mnist.data import get_dataloaders
from mnist.evaluate import evaluate
from mnist.type_aliases import CrossEntropyLossFunction
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
    loss_fn = cast(CrossEntropyLossFunction, torch.nn.CrossEntropyLoss())

    # W&B plumbing
    name = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = wandb.init(project="mnist", name=name, config=config.model_dump())
    wandb.define_metric("step_number", hidden=True)
    wandb.define_metric("train/loss", step_metric="step_number")
    wandb.define_metric("validation/loss", step_metric="step_number")
    wandb.define_metric("validation/accuracy", step_metric="step_number")

    data = get_dataloaders(config.data)

    epoch_count = config.train.epoch_count
    step_number = 0
    for epoch in range(epoch_count):
        for batch in data.train:
            # We could also have copied this all over upfront for MNIST, since it's small
            batch_images, batch_labels = batch
            images: torch.Tensor = batch_images.to(device)
            labels: torch.Tensor = batch_labels.to(device)

            # Compute loss
            logits = model(images)
            train_loss = loss_fn(logits, labels)

            # Zero the gradients for this step
            optimizer.zero_grad()

            # Backprop the gradients through the computation graph
            train_loss.backward()

            # Update weights using the computed gradients
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]

            # Log training loss every Nth batch/step
            step_number += 1
            if step_number % 10 == 0:
                # A sync point explicitly moving data from the GPU to the CPU
                train_loss_value = train_loss.item()
                wandb.log({"train/loss": train_loss_value, "step_number": step_number})
        
        with torch.no_grad():
            # Log validation loss once per epoch
            validation_loss, validation_accuracy = evaluate(model, data.validation, device, loss_fn)
            wandb.log({"validation/loss": validation_loss, "validation/accuracy": validation_accuracy, "step_number": step_number})

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{run.id}.pt"
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    main()
