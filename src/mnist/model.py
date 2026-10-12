from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

"""
LeNet-5 layer spec (batch dim omitted; shapes are C x H x W)

Layer   Operation                            Output shape
------  -----------------------------------  ------------
Input   grayscale image (28x28, padded to 32) 1x32x32
C1      conv, 6 filters, 5x5                 6x28x28
S2      avg pool 2x2, stride 2               6x14x14
C3      conv, 16 filters, 5x5                16x10x10
S4      avg pool 2x2, stride 2               16x5x5
C5      conv 5x5 -> 120 ch (= flatten 400    120
        -> 120 + linear)
F6      linear, 120 -> 84                    84
Output  linear, 84 -> 10                     10 logits

Total params ~61.7k (full C3 connectivity)
"""

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 16*5*5 comes from the final shape of the most recent layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x: Tensor) -> Tensor:
        # C1
        x = F.relu(self.conv1(x))
        # S2
        x = F.avg_pool2d(x, 2, 2)
        # C3
        x = F.relu(self.conv2(x))
        # S4
        x = F.avg_pool2d(x, 2, 2)
        # C5
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        # F6
        x = F.relu(self.fc2(x))
        # Output
        # No nonlinearity after the last layer.
        x = self.fc3(x)
        return x