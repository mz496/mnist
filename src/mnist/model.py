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

# TODO 10-09: This needs the nonlinearities.

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

    def forward(self, x):
        print("start")
        print(x.shape)

        x = nn.Conv2d(1, 6, 5, padding=2)(x)
        print("C1 done")
        print(x.shape)

        x = nn.AvgPool2d(2, 2)(x)
        print("S2 done")
        print(x.shape)

        x = nn.Conv2d(6, 16, 5)(x)
        print("C3 done")
        print(x.shape)

        x = nn.AvgPool2d(2, 2)(x)
        print("S4 done")
        print(x.shape)

        x = nn.Flatten()(x)
        # 16*5*5 comes from the final shape of the most recent layer
        x = nn.Linear(16 * 5 * 5, 120)(x)
        print("C5 done")
        print(x.shape)

        x = nn.Linear(120, 84)(x)
        print("F6 done")
        print(x.shape)

        x = nn.Linear(84, 10)(x)
        print("final")
        print(x.shape)

        return x