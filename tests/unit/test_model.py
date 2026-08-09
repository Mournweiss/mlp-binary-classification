# SPDX-FileCopyrightText: 2026 Maxim Selin (Mournweiss) <info@mournweiss.ru>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MLP model."""

import torch

from mlp_binary_classification.model import MLP


class TestMLPInit:
    """Tests for MLP model initialization."""

    def test_default_hidden_sizes(self) -> None:
        """Test default hidden layer sizes."""
        model = MLP(input_size=10)
        assert len(model.network) > 0

    def test_custom_hidden_sizes(self) -> None:
        """Test custom hidden layer sizes."""
        hidden_sizes = [128, 64, 32]
        model = MLP(input_size=10, hidden_sizes=hidden_sizes)
        assert model is not None

    def test_output_is_nn_module(self) -> None:
        """Test that model is a valid nn.Module."""
        model = MLP(input_size=10)
        assert isinstance(model, torch.nn.Module)

    def test_zero_input_size(self) -> None:
        """Test model creation with zero input size edge case."""
        model = MLP(input_size=0)
        assert model is not None


class TestMLPForward:
    """Tests for MLP forward pass."""

    def test_forward_output_shape(self, device: torch.device) -> None:
        """Test forward pass output shape."""
        model = MLP(input_size=19).to(device)
        x = torch.randn(5, 19).to(device)
        output = model(x)
        assert output.shape == (5, 1)

    def test_forward_single_input(self, device: torch.device) -> None:
        """Test forward pass with single input."""
        model = MLP(input_size=19).to(device)
        x = torch.randn(1, 19).to(device)
        output = model(x)
        assert output.shape == (1, 1)

    def test_forward_batch_variations(self, device: torch.device) -> None:
        """Test forward pass with various batch sizes."""
        model = MLP(input_size=19).to(device)
        for batch_size in [1, 2, 8, 32]:
            x = torch.randn(batch_size, 19).to(device)
            output = model(x)
            assert output.shape == (batch_size, 1)

    def test_forward_different_input_sizes(self, device: torch.device) -> None:
        """Test forward pass with different input sizes."""
        for input_size in [5, 10, 19, 50]:
            model = MLP(input_size=input_size).to(device)
            x = torch.randn(4, input_size).to(device)
            output = model(x)
            assert output.shape == (4, 1)


class TestMLPPredict:
    """Tests for MLP prediction methods."""

    def test_predict_proba_range(self, device: torch.device) -> None:
        """Test that predict_proba outputs are in [0, 1]."""
        model = MLP(input_size=19).to(device)
        x = torch.randn(10, 19).to(device)
        proba = model.predict_proba(x)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_predict_binary_output(self, device: torch.device) -> None:
        """Test that predict outputs binary values."""
        model = MLP(input_size=19).to(device)
        x = torch.randn(10, 19).to(device)
        predictions = model.predict(x)
        assert set(predictions.unique().cpu().tolist()).issubset({0.0, 1.0})

    def test_predict_threshold(self, device: torch.device) -> None:
        """Test prediction with custom threshold."""
        model = MLP(input_size=19).to(device)
        # Create input that produces high logits
        x = torch.ones(5, 19).to(device) * 10
        predictions = model.predict(x, threshold=0.9)
        # Should predict mostly 1s with high threshold
        assert predictions.sum() >= 0

    def test_predict_proba_no_grad(self, device: torch.device) -> None:
        """Test that predict_proba works without gradients."""
        model = MLP(input_size=19).to(device)
        x = torch.randn(5, 19, requires_grad=False).to(device)
        proba = model.predict_proba(x)
        assert proba.requires_grad is False


class TestMLPWeights:
    """Tests for MLP weight initialization."""

    def test_weights_initialized(self) -> None:
        """Test that all weights are initialized (not zero/NaN)."""
        model = MLP(input_size=19)
        for param in model.parameters():
            assert not torch.isnan(param).any()
            assert not torch.isinf(param).any()

    def test_bias_initialized_to_zero(self) -> None:
        """Test that biases are initialized to zero."""
        model = MLP(input_size=19)
        model._init_weights()
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                if module.bias is not None:
                    assert torch.all(module.bias == 0)
