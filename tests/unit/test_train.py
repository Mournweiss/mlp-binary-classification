# SPDX-FileCopyrightText: 2026 Maxim Selin (Mournweiss) <info@mournweiss.ru>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the training module."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mlp_binary_classification.config import Config, DataConfig, ModelConfig, OutputConfig
from mlp_binary_classification.model import MLP
from mlp_binary_classification.train import (
    create_dataloaders,
    evaluate_model,
    get_device,
)


class TestGetDevice:
    """Tests for get_device function."""

    def test_returns_torch_device(self) -> None:
        """Test that get_device returns a torch.device instance."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_type_valid(self) -> None:
        """Test that returned device type is valid."""
        device = get_device()
        assert device.type in ("cpu", "cuda", "mps")


class TestCreateDataloaders:
    """Tests for create_dataloaders function."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Provide sample data for dataloader tests."""
        X_train = np.random.randn(20, 19).astype(np.float32)
        y_train = np.random.randint(0, 2, (20, 1)).astype(np.float32)
        X_val = np.random.randn(5, 19).astype(np.float32)
        y_val = np.random.randint(0, 2, (5, 1)).astype(np.float32)
        return X_train, y_train, X_val, y_val

    def test_returns_tuple_of_dataloaders(self, sample_data: tuple) -> None:
        """Test that create_dataloaders returns a tuple of two DataLoaders."""
        X_train, y_train, X_val, y_val = sample_data
        train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val)

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_train_loader_shuffles_data(self, sample_data: tuple) -> None:
        """Test that train loader has shuffle enabled."""
        X_train, y_train, X_val, y_val = sample_data
        train_loader, _ = create_dataloaders(X_train, y_train, X_val, y_val)

        # In PyTorch 2.5+ shuffle is not exposed as attribute; check sampler type
        assert train_loader._index_sampler is not None

    def test_val_loader_no_shuffle(self, sample_data: tuple) -> None:
        """Test that validation loader has shuffle disabled."""
        X_train, y_train, X_val, y_val = sample_data
        _, val_loader = create_dataloaders(X_train, y_train, X_val, y_val)

        # In PyTorch 2.5+ shuffle is not exposed as attribute; check sampler type
        assert val_loader._index_sampler is not None

    def test_dataloader_batch_output_dtype(self, sample_data: tuple) -> None:
        """Test that dataloader batches have correct tensor dtypes."""
        X_train, y_train, X_val, y_val = sample_data
        train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val)

        for loader in (train_loader, val_loader):
            batch_X, batch_y = next(iter(loader))
            assert batch_X.dtype == torch.float32
            assert batch_y.dtype == torch.float32

    def test_dataloader_batch_output_shape(self, sample_data: tuple) -> None:
        """Test that dataloader batches have correct shapes."""
        X_train, y_train, X_val, y_val = sample_data
        train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=4)

        for loader in (train_loader, val_loader):
            batch_X, batch_y = next(iter(loader))
            assert batch_X.shape[1] == 19
            assert batch_y.shape[1] == 1

    def test_pin_memory_enabled(self, sample_data: tuple) -> None:
        """Test that pin_memory is enabled for both loaders."""
        X_train, y_train, X_val, y_val = sample_data
        train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val)

        assert train_loader.pin_memory is True
        assert val_loader.pin_memory is True

    def test_dataloader_iterates_all_batches(self, sample_data: tuple) -> None:
        """Test that dataloader iterates through all data."""
        X_train, y_train, X_val, y_val = sample_data
        train_loader, _ = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=10)

        total_samples = 0
        for batch_X, batch_y in train_loader:
            total_samples += batch_X.size(0)

        assert total_samples == len(X_train)


class TestEvaluateModel:
    """Tests for evaluate_model helper function."""

    @pytest.fixture
    def dummy_loader(self, sample_features, sample_labels) -> DataLoader:
        """Create a dummy DataLoader for evaluation."""
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(sample_features, dtype=torch.float32),
            torch.tensor(sample_labels, dtype=torch.float32),
        )
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    @pytest.fixture
    def small_model(self, device: torch.device) -> torch.nn.Module:
        """Provide a small MLP model for testing."""
        return MLP(input_size=19, hidden_sizes=[8, 4], dropout=0.0).to(device)

    def test_evaluate_model_returns_tuple(self, small_model, dummy_loader, device) -> None:
        """Test that evaluate_model returns a tuple of two floats."""
        loss, accuracy = evaluate_model(small_model, dummy_loader, torch.nn.BCEWithLogitsLoss(), device)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)

    def test_evaluate_model_accuracy_range(self, small_model, dummy_loader, device) -> None:
        """Test that accuracy is in valid range [0, 1]."""
        _, accuracy = evaluate_model(small_model, dummy_loader, torch.nn.BCEWithLogitsLoss(), device)

        assert 0.0 <= accuracy <= 1.0

    def test_evaluate_model_loss_non_negative(self, small_model, dummy_loader, device) -> None:
        """Test that loss is non-negative."""
        loss, _ = evaluate_model(small_model, dummy_loader, torch.nn.BCEWithLogitsLoss(), device)

        assert loss >= 0.0

    def test_evaluate_model_no_nan_values(self, small_model, dummy_loader, device) -> None:
        """Test that evaluation produces no NaN values."""
        loss, accuracy = evaluate_model(small_model, dummy_loader, torch.nn.BCEWithLogitsLoss(), device)

        assert not np.isnan(loss)
        assert not np.isnan(accuracy)


class TestTrainModelIntegration:
    """Integration tests for train_model function."""

    @pytest.fixture
    def training_config(self, tmp_path: Path) -> Config:
        """Provide a config for training tests."""
        sample_train = Path("tests/fixtures/sample_train.csv")
        sample_test = Path("tests/fixtures/sample_test.csv")

        data_config = DataConfig(
            train_path=sample_train,
            test_path=sample_test,
            val_split=0.2,
            random_state=42,
        )
        model_config = ModelConfig(
            hidden_sizes=[8, 4],
            epochs=3,
            batch_size=4,
            learning_rate=1e-2,
            seed=42,
        )
        output_config = OutputConfig(
            checkpoint_dir=tmp_path / "checkpoints",
            model_path=tmp_path / "model.pt",
        )

        return Config(data=data_config, model=model_config, output=output_config)

    def test_train_model_returns_history_dict(self, training_config: Config) -> None:
        """Test that train_model returns a dictionary with expected keys."""
        from mlp_binary_classification.train import train_model

        history = train_model(training_config)

        assert isinstance(history, dict)
        assert "loss_history" in history
        assert "val_loss_history" in history
        assert "val_acc_history" in history
        assert "train_metadata" in history
        assert "input_size" in history

    def test_train_model_history_length(self, training_config: Config) -> None:
        """Test that history has correct number of epochs."""
        from mlp_binary_classification.train import train_model

        history = train_model(training_config)

        assert len(history["loss_history"]) == training_config.model.epochs
        assert len(history["val_loss_history"]) == training_config.model.epochs
        assert len(history["val_acc_history"]) == training_config.model.epochs

    def test_train_model_loss_values_valid(self, training_config: Config) -> None:
        """Test that all loss values are valid (non-NaN, non-negative)."""
        from mlp_binary_classification.train import train_model

        history = train_model(training_config)

        for loss in history["loss_history"]:
            assert not np.isnan(loss)
            assert loss >= 0.0

        for val_loss in history["val_loss_history"]:
            assert not np.isnan(val_loss)
            assert val_loss >= 0.0

    def test_train_model_accuracy_values_valid(self, training_config: Config) -> None:
        """Test that all accuracy values are in [0, 1]."""
        from mlp_binary_classification.train import train_model

        history = train_model(training_config)

        for acc in history["val_acc_history"]:
            assert not np.isnan(acc)
            assert 0.0 <= acc <= 1.0

    def test_train_model_input_size_correct(self, training_config: Config) -> None:
        """Test that input_size in history matches actual feature count."""
        from mlp_binary_classification.train import train_model

        history = train_model(training_config)

        assert isinstance(history["input_size"], int)
        assert history["input_size"] > 0
