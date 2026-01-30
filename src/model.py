"""PyTorch model definitions."""

from typing import Any

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for classification.

    Architecture: input → linear → relu → dropout → linear → output
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize SimpleMLP.

        Args:
            input_dim: Input feature dimension
            hidden_size: Hidden layer size
            num_layers: Number of hidden layers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()

        layers: list[Any] = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_size, num_classes))

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # type: ignore
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)


def create_model(
    input_dim: int = 10,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_classes: int = 2,
    dropout: float = 0.1,
    device: str = "cpu",
) -> SimpleMLP:
    """Create and initialize model.

    Args:
        input_dim: Input feature dimension
        hidden_size: Hidden layer size
        num_layers: Number of hidden layers
        num_classes: Number of output classes
        dropout: Dropout probability
        device: Device to place model on (cpu, mps, cuda)

    Returns:
        Model instance on specified device
    """
    model = SimpleMLP(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )
    model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
