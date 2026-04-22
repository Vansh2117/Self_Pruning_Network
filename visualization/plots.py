"""
visualization/plots.py
----------------------
All matplotlib visualisations for the self-pruning experiment.

plot_gate_histograms  : distribution of final gate values per lambda
plot_training_curves  : CE loss and sparsity loss over training epochs
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from model.prunable_layer import PrunableLinear


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _collect_gates(model: torch.nn.Module) -> np.ndarray:
    """Return all gate values from every PrunableLinear layer as a 1-D array."""
    parts = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                parts.append(gates.cpu().numpy().ravel())
    return np.concatenate(parts) if parts else np.array([])


# ──────────────────────────────────────────────────────────────────────────────
# Public plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_gate_histograms(
    models_dict: dict[str, torch.nn.Module],
    save_path:   str = "results/gate_histograms.png",
    threshold:   float = 1e-2,
) -> None:
    """
    One histogram per model showing the distribution of gate values after
    training.

    Expected signature for a well-pruned model:
      • Large spike near 0  → pruned connections
      • Smaller cluster near 1 → active connections
      • Thin (or empty) middle region

    Parameters
    ----------
    models_dict : {label: trained SelfPruningNet}, e.g. {"λ=0.001": model}
    save_path   : output file path (PNG)
    threshold   : dashed line marking the pruning threshold
    """
    n      = len(models_dict)
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (label, model), color in zip(axes, models_dict.items(), colors):
        gates = _collect_gates(model)
        if gates.size == 0:
            ax.set_title(f"{label}\n(no PrunableLinear found)")
            continue

        pct_pruned = 100.0 * (gates < threshold).sum() / len(gates)

        ax.hist(
            gates, bins=80,
            color=color, alpha=0.85,
            edgecolor="white", linewidth=0.3,
        )
        ax.axvline(
            x=threshold, color="black",
            linestyle="--", linewidth=1.4,
            label=f"Threshold ({threshold})",
        )
        ax.set_title(f"Gate Distribution — {label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Gate value  g = σ(s)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=8)
        ax.text(
            0.62, 0.88,
            f"Pruned: {pct_pruned:.1f}%",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    fig.suptitle(
        "Self-Pruning Network: Final Gate Value Distributions",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Gate histograms saved → {save_path}")


def plot_training_curves(
    histories: dict[str, dict[str, list[float]]],
    save_path: str = "results/training_curves.png",
) -> None:
    """
    Plot CE loss and raw sparsity loss over training epochs for every run.

    Parameters
    ----------
    histories : {label: history_dict} where history_dict contains
                "ce_loss" and "sparsity_loss" lists
    save_path : output file path (PNG)
    """
    palette = {
        "Baseline": "#2ecc71",
        "λ=0.001":  "#3498db",
        "λ=0.01":   "#e67e22",
        "λ=0.1":    "#e74c3c",
    }
    default_colors = ["#9b59b6", "#1abc9c", "#e91e63"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for idx, (label, hist) in enumerate(histories.items()):
        color  = palette.get(label, default_colors[idx % len(default_colors)])
        style  = "--" if label == "Baseline" else "-"
        epochs = range(1, len(hist["ce_loss"]) + 1)

        axes[0].plot(epochs, hist["ce_loss"],
                     label=label, color=color, linestyle=style, linewidth=1.8)
        axes[1].plot(epochs, hist["sparsity_loss"],
                     label=label, color=color, linestyle=style, linewidth=1.8)

    for ax, title, ylabel in zip(
        axes,
        ["Classification Loss (CE) per Epoch",
         "Avg Sparsity Loss per Epoch\n(Σ gate values, before λ scaling)"],
        ["Cross-Entropy Loss", "Σ sigmoid(s)"],
    ):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves saved → {save_path}")
