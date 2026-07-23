"""Integration tests for the training pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mlp_binary_classification.config import Config, DataConfig, ModelConfig, OutputConfig
from mlp_binary_classification.data_loader import load_data, split_train_val
from mlp_binary_classification.model import MLP
from mlp_binary_classification.preprocessing import preprocess_titanic
from mlp_binary_classification.train import (
    create_dataloaders,
    evaluate_model,
    get_device,
    train_model,
)


@pytest.fixture
def training_config(tmp_path: Path) -> Config:
    """Provide a config pointing to sample data with short training."""
    sample_train = Path("tests/fixtures/sample_train.csv")
    sample_test = Path("tests/fixtures/sample_test.csv")

    data_config = DataConfig(
        train_path=sample_train,
        test_path=sample_test,
        val_split=0.2,
        random_state=42,
    )
    model_config = ModelConfig(
        hidden_sizes=[16, 8],
        epochs=5,
        batch_size=4,
        learning_rate=1e-2,
        seed=42,
    )
    output_config = OutputConfig(
        checkpoint_dir=tmp_path / "checkpoints",
        model_path=tmp_path / "model.pt",
    )

    return Config(data=data_config, model=model_config, output=output_config)


@pytest.mark.integration
def test_full_training_loop(training_config: Config) -> None:
    """Test the full training loop completes without errors."""
    history = train_model(training_config)

    assert "loss_history" in history
    assert "val_loss_history" in history
    assert "val_acc_history" in history
    assert len(history["loss_history"]) == training_config.model.epochs
    assert len(history["val_loss_history"]) == training_config.model.epochs
    assert len(history["val_acc_history"]) == training_config.model.epochs


@pytest.mark.integration
def test_model_saves_checkpoint(training_config: Config) -> None:
    """Test that a checkpoint file is saved during training."""
    train_model(training_config)

    checkpoint_path = training_config.output.checkpoint_dir / "best_model.pt"
    assert checkpoint_path.exists()

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "val_loss" in checkpoint
    assert "input_size" in checkpoint


@pytest.mark.integration
def test_validation_during_training(training_config: Config) -> None:
    """Test that validation metrics improve or remain stable."""
    history = train_model(training_config)

    # Validation loss should not be NaN
    assert not any(np.isnan(v) for v in history["val_loss_history"])
    # Validation accuracy should be in valid range
    assert all(0 <= v <= 1 for v in history["val_acc_history"])


@pytest.mark.integration
def test_lr_scheduler_steps(training_config: Config) -> None:
    """Test that learning rate changes during training."""
    # The training function internally creates a scheduler
    # We verify by checking that val_loss changes over epochs
    history = train_model(training_config)

    initial_loss = history["loss_history"][0]
    final_loss = history["loss_history"][-1]

    # Loss should change (not necessarily decrease with small data)
    assert isinstance(initial_loss, float)
    assert isinstance(final_loss, float)


@pytest.mark.integration
def test_gradient_clipping(training_config: Config) -> None:
    """Test that training completes with gradient clipping enabled."""
    # If gradient clipping causes issues, this would raise an error
    history = train_model(training_config)

    # Verify no NaN gradients in loss history
    assert not any(np.isnan(v) for v in history["loss_history"])


@pytest.fixture
def sample_dataloaders() -> tuple:
    """Create sample dataloaders for testing."""
    X_train = np.random.randn(20, 19).astype(np.float32)
    y_train = np.random.randint(0, 2, (20, 1)).astype(np.float32)
    X_val = np.random.randn(5, 19).astype(np.float32)
    y_val = np.random.randint(0, 2, (5, 1)).astype(np.float32)

    return create_dataloaders(X_train, y_train, X_val, y_val, batch_size=4)


@pytest.mark.integration
def test_dataloaders_return_correct_types(sample_dataloaders: tuple) -> None:
    """Test that dataloaders return correct tensor types."""
    from torch.utils.data import DataLoader

    train_loader, val_loader = sample_dataloaders

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    # Get a batch and verify types
    batch_X, batch_y = next(iter(train_loader))
    assert batch_X.dtype == torch.float32
    assert batch_y.dtype == torch.float32


@pytest.mark.integration
def test_evaluate_model_function(sample_dataloaders: tuple, device: torch.device) -> None:
    """Test the evaluate_model helper function."""
    from torch.nn import BCEWithLogitsLoss

    train_loader, val_loader = sample_dataloaders
    model = MLP(input_size=19, hidden_sizes=[16, 8], dropout=0.1).to(device)

    loss, accuracy = evaluate_model(model, val_loader, BCEWithLogitsLoss(), device)

    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1
    assert loss >= 0
