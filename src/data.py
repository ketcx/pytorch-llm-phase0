"""PyTorch data utilities."""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    """Dummy dataset for testing.

    Generates random samples with targets for demonstration.
    """

    def __init__(self, size: int = 1000, input_dim: int = 10, num_classes: int = 2):
        """Initialize dummy dataset.

        Args:
            size: Number of samples
            input_dim: Input feature dimension
            num_classes: Number of classes
        """
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Generate dummy data
        self.X = np.random.randn(size, input_dim).astype(np.float32)
        self.y = np.random.randint(0, num_classes, size).astype(np.int64)

    def __len__(self) -> int:
        """Return dataset size."""
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input tensor, target tensor)
        """
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def create_dataloaders(
    dataset_size: int = 1000,
    input_dim: int = 10,
    num_classes: int = 2,
    batch_size: int = 32,
    num_workers: int = 0,
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders.

    Args:
        dataset_size: Total dataset size
        input_dim: Input feature dimension
        num_classes: Number of classes
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        split: (train, val, test) split ratios

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    dataset = DummyDataset(size=dataset_size, input_dim=input_dim, num_classes=num_classes)

    # Split indices
    train_size = int(len(dataset) * split[0])
    val_size = int(len(dataset) * split[1])

    indices = torch.randperm(len(dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create subsets
    from torch.utils.data import Subset

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
