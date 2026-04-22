"""
training/trainer.py

Training loop and evaluation utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.loss import total_loss
from model.prunable_layer import PrunableLinear


def train_model(model, loader, lambda_, device, epochs=20, lr=1e-3, prunable=True):
    """
    Full training loop. Returns loss history dict for plotting.

    prunable=False skips the sparsity term (used for baseline).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"ce_loss": [], "sparsity_loss": [], "total_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        sum_ce, sum_sp, sum_total, n = 0.0, 0.0, 0.0, 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)

            if prunable:
                loss, ce, sp = total_loss(logits, labels, model, lambda_)
            else:
                ce   = F.cross_entropy(logits, labels)
                loss = ce
                sp   = torch.tensor(0.0)

            loss.backward()
            optimizer.step()

            sum_ce    += ce.item()
            sum_sp    += sp.item()
            sum_total += loss.item()
            n         += 1

        # store epoch averages
        history["ce_loss"].append(sum_ce / n)
        history["sparsity_loss"].append(sum_sp / n)
        history["total_loss"].append(sum_total / n)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"    Epoch {epoch:02d}/{epochs} | "
                f"CE: {sum_ce/n:.4f} | "
                f"Sparsity: {sum_sp/n:.1f} | "
                f"Total: {sum_total/n:.4f}"
            )

    return history


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds    = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return 100.0 * correct / total


def compute_sparsity_pct(model, threshold=1e-2):
    """
    Fraction of gates below threshold, as a percentage.
    Gate < 0.01 means the weight contributes less than 1% -- effectively pruned.
    """
    pruned, total = 0, 0

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates   = torch.sigmoid(module.gate_scores)
                pruned += (gates < threshold).sum().item()
                total  += gates.numel()

    return 100.0 * pruned / total if total > 0 else 0.0
