"""
train.py -- entry point

Trains a self-pruning MLP on CIFAR-10 and compares results
across different sparsity regularization strengths (lambda).

Usage:
    python train.py                         # defaults
    python train.py --epochs 25
    python train.py --epochs 30 --lambdas 0.0001 0.001 0.01 0.1
"""

import argparse
import os
import torch
import numpy as np

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

from data import get_dataloaders
from model import SelfPruningNet, BaselineNet
from training  import train_model, evaluate_accuracy, compute_sparsity_pct
from visualization import plot_gate_histograms, plot_training_curves


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=25)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--lambdas",    type=float, nargs="+", default=[50.0, 100.0, 150.0])
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print(f"  Self-Pruning Network | device: {device} | epochs: {args.epochs}")
    print("=" * 60)

    # -- data --
    print("\nLoading CIFAR-10...")
    train_loader, test_loader = get_dataloaders(args.batch_size)

    # -- baseline --
    print("\n[Baseline] Training standard MLP (no pruning)...")
    baseline = BaselineNet().to(device)
    baseline_hist = train_model(
        baseline, train_loader, lambda_=0.0,
        device=device, epochs=args.epochs, lr=args.lr, prunable=False
    )
    baseline_acc = evaluate_accuracy(baseline, test_loader, device)
    print(f"  Baseline accuracy: {baseline_acc:.2f}%\n")

    # -- pruning runs --
    results        = {}
    trained_models = {}
    all_histories  = {"Baseline": baseline_hist}

    for lam in args.lambdas:
        label = f"λ={lam}"
        print(f"[{label}] Training...")

        model = SelfPruningNet().to(device)
        hist  = train_model(
            model, train_loader, lambda_=lam,
            device=device, epochs=args.epochs, lr=args.lr, prunable=True
        )

        acc = evaluate_accuracy(model, test_loader, device)
        sp  = compute_sparsity_pct(model)

        results[lam]          = {"accuracy": acc, "sparsity": sp}
        trained_models[label] = model
        all_histories[label]  = hist

        print(f"  accuracy: {acc:.2f}% | sparsity: {sp:.2f}%\n")

    # -- results table --
    print("=" * 60)
    print(f"  {'Model':<12} {'Accuracy':>12} {'Sparsity':>12}")
    print("  " + "-" * 38)
    print(f"  {'Baseline':<12} {baseline_acc:>11.2f}%  {'N/A':>10}")
    for lam in args.lambdas:
        r = results[lam]
        print(f"  {f'λ={lam}':<12} {r['accuracy']:>11.2f}%  {r['sparsity']:>9.2f}%")
    print("=" * 60)

    # -- plots --
    plot_gate_histograms(trained_models, save_path="results/gate_histograms.png")
    plot_training_curves(all_histories,  save_path="results/training_curves.png")
    print("\nPlots saved to ./results/")


if __name__ == "__main__":
    main()
