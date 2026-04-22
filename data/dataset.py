"""
data/dataset.py
---------------
Loads CIFAR-10 from torchvision, applies standard normalization, and wraps
it in PyTorch DataLoaders for training and evaluation.

Normalization statistics (mean / std per channel) are the widely-used
per-channel values computed over the CIFAR-10 training set:
    R: mean=0.4914, std=0.2470
    G: mean=0.4822, std=0.2435
    B: mean=0.4465, std=0.2616
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloaders(
    batch_size: int = 128,
    data_root:  str = "./data",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Download (if needed) and return CIFAR-10 train and test DataLoaders.

    Parameters
    ----------
    batch_size  : mini-batch size for both loaders
    data_root   : directory where the dataset will be cached
    num_workers : parallel workers for data loading

    Returns
    -------
    (train_loader, test_loader)
    """
    # ── Transforms ────────────────────────────────────────────────────────
    # ToTensor()  converts PIL Image [0,255] → float Tensor [0.0, 1.0]
    # Normalize() applies (x - mean) / std channel-wise
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std =(0.2470, 0.2435, 0.2616),
        ),
    ])

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True,  download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # shuffle every epoch during training
        num_workers=num_workers,
        pin_memory=True,        # speeds up host→GPU transfer
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"[Dataset] CIFAR-10 loaded — "
        f"train: {len(train_dataset):,} samples | "
        f"test: {len(test_dataset):,} samples | "
        f"batch size: {batch_size}"
    )
    return train_loader, test_loader
