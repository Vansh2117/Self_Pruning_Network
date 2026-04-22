"""
model/network.py

Two models for comparison:
  SelfPruningNet  -- uses PrunableLinear, learns to drop connections
  BaselineNet     -- standard MLP with dropout, used as reference

CIFAR-10 images are 3x32x32 = 3072 features when flattened.
Architecture: 3072 -> 512 -> 256 -> 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prunable_layer import PrunableLinear


class SelfPruningNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten: (B, 3, 32, 32) -> (B, 3072)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)           # raw logits, no softmax

    def prunable_layers(self):
        """Returns all PrunableLinear layers -- used by the loss function."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]


class BaselineNet(nn.Module):
    """
    Same architecture as SelfPruningNet but with standard nn.Linear + dropout.
    Dropout prevents the baseline from overfitting, making the accuracy
    comparison against the pruned models fair.
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        self.fc1     = nn.Linear(3072, 512)
        self.fc2     = nn.Linear(512, 256)
        self.fc3     = nn.Linear(256, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)
