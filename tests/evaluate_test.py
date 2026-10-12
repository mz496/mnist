import torch
from mnist.evaluate import count_correct_in_batch

def test_counts_matching_predictions():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])  # argmax: 1, 0, 1
    labels = torch.tensor([1, 0, 0])
    assert count_correct_in_batch(logits, labels) == 2