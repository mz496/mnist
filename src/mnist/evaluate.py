from typing import Any, Callable
from torch import Tensor, nn
import torch
from torch.utils.data import DataLoader

from mnist.data import InputDatasetItem
from mnist.type_aliases import CrossEntropyLossFunction

def count_correct_in_batch(logits: Tensor, labels: Tensor) -> int:
    """
    Returns the number of correct predictions.
    logits: [B, 10]
    labels: [B]
    """
    if len(logits) != len(labels):
        raise Exception("logits and labels must be the same batch size")
    # Collapse [B, 10] into [B], each row containing a prediction in the batch
    preds = logits.argmax(dim=1)
    # Compare against [B] labels
    is_pred_correct = preds == labels
    # Sum where True -> 1
    return int(is_pred_correct.sum().item())

def evaluate(model: nn.Module, dataloader: DataLoader[InputDatasetItem], device: str, loss_fn: CrossEntropyLossFunction):
    # Set eval mode on the model
    model.eval()

    total_loss = 0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_images, batch_labels = batch
            images = batch_images.to(device)
            # [B] of numbers from 0-9
            labels = batch_labels.to(device)
            n = images.size(0)

            # [B, 10], one row of predictions of all classes
            logits = model(images)
            # The average loss for this batch
            loss = loss_fn(logits, labels)
            
            # The total loss for all examples in this batch
            batch_total_loss = loss.item() * n
            total_loss += batch_total_loss
            total_correct += count_correct_in_batch(logits, labels)
            total_examples += n
    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    print("validation check: total examples", total_examples)

    return average_loss, accuracy