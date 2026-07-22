"""MLP model definition for binary classification."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron for binary classification.

    Architecture:
        Input -> Linear -> ReLU -> Dropout -> ... -> Linear -> Sigmoid
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = None,
        dropout: float = 0.3,
    ) -> None:
        """Initialize MLP model.

        Args:
            input_size: Number of input features.
            hidden_sizes: List of hidden layer sizes.
            dropout: Dropout probability.
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32, 16]

        layers: list[nn.Module] = []
        in_features = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1:  # No dropout after last hidden layer
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size

        # Output layer (no activation — will use BCEWithLogitsLoss)
        layers.append(nn.Linear(in_features, 1))

        self.network = nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming uniform distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output logits of shape (batch_size, 1).
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probabilities using sigmoid.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Probabilities of shape (batch_size, 1).
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict class labels.

        Args:
            x: Input tensor of shape (batch_size, input_size).
            threshold: Classification threshold.

        Returns:
            Predicted labels of shape (batch_size,).
        """
        proba = self.predict_proba(x)
        return (proba >= threshold).float().squeeze(-1)
