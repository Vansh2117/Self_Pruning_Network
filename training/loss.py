"""
training/loss.py

Total loss = CrossEntropy + lambda * sparsity_loss

The sparsity loss is the *mean* gate value across all gates
(not the sum). This normalization matters: with ~1.7 million gates,
the raw sum at initialization is ~850,000 -- which would dwarf the
CE loss (~1.6) even at lambda=0.001. By dividing by the gate count,
the sparsity term stays on the same scale as CE regardless of model
size, and lambda values have intuitive meaning (0.1 = "I'm willing
to trade 0.1 nats of CE loss for a 1-unit reduction in mean gate").

Why mean gate values drive sparsity:
  sigmoid(s) is always positive, so penalizing its mean pushes the
  optimizer to drive gate scores negative. As s -> -inf, sigmoid(s) -> 0
  and that weight is effectively removed. The classification loss
  resists this for connections it needs -- so only genuinely redundant
  connections get pruned.
"""

import torch
import torch.nn.functional as F
from model.prunable_layer import PrunableLinear


def sparsity_loss(model):
    """
    Mean gate value across all PrunableLinear layers.
    Using mean instead of sum keeps this on the same scale as CE loss,
    so lambda values are comparable across different model sizes.
    """
    device = next(model.parameters()).device
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            all_gates.append(torch.sigmoid(module.gate_scores).ravel())

    if not all_gates:
        return torch.tensor(0.0, device=device)

    # cat all gates into one vector, take the mean
    return torch.cat(all_gates).mean()


def total_loss(logits, labels, model, lambda_):
    """
    Combined loss. Returns (total, ce, sparsity) for logging.
    F.cross_entropy expects raw logits -- applies log-softmax internally.
    """
    ce = F.cross_entropy(logits, labels)
    sp = sparsity_loss(model)
    return ce + lambda_ * sp, ce, sp
