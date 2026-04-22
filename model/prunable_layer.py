"""
model/prunable_layer.py

Custom linear layer with a learnable gate on each weight.
The idea: instead of pruning after training, let the network
decide which connections to keep *during* training.

Each weight w has a paired gate score s.
  gate = sigmoid(s)        -- squashes to (0, 1)
  effective_w = w * gate   -- zero gate = pruned connection
  output = input @ effective_w.T + bias

Gradients flow through both w and s automatically since both
are nn.Parameter. The sparsity loss (in loss.py) pushes s toward
-inf during training, which collapses the gate toward 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Standard weight + bias, same as any linear layer
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # One gate score per weight element -- same shape as weight.
        # Initialized randomly around 0 with small noise.
        # Why not zeros? If all gate scores start identical, every gate
        # receives the same gradient from the sparsity loss and they all
        # move together -- the network can't learn *which* connections
        # to keep vs prune. Random init breaks that symmetry.
        # sigmoid(~0) ≈ 0.5 so gates still start roughly half-open.
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        # Kaiming uniform for weights -- standard for ReLU networks.
        # Keeps activation variance stable across layers.
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x):
        # Compute gate values: sigmoid maps scores to (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Mask the weights -- connections with gate near 0 contribute nothing
        pruned_w = self.weight * gates

        # Standard linear op using the masked weights
        return F.linear(x, pruned_w, self.bias)
