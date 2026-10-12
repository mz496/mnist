from typing import Any, Callable
from torch import Tensor, nn
from torch.utils.data import DataLoader

from mnist.data import InputDatasetItem

def count_correct_in_batch(logits: Tensor, labels: Tensor):
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
    return is_pred_correct.sum()

def evaluate(model: nn.Module, dataloader: DataLoader[InputDatasetItem], device: str, loss_fn: Callable[[Any, Any], float]):
    total_loss = 0
    total_correct = 0
    for batch in dataloader:
        batch_images, batch_labels = batch
        images = batch_images.to(device)
        # [B] of numbers from 0-9
        labels = batch_labels.to(device)

        # [B, 10], one row of predictions of all classes
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += loss
        total_correct += count_correct_in_batch(logits, labels)
    average_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader)

    return average_loss, accuracy