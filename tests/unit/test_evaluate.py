"""Unit tests for the evaluation module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mlp_binary_classification.config import Config, DataConfig, ModelConfig, OutputConfig
from mlp_binary_classification.evaluate import (
    evaluate_on_test,
    evaluate_on_validation,
)
from mlp_binary_classification.model import MLP


class TestEvaluateOnTest:
    """Tests for evaluate_on_test function."""

    @pytest.fixture
    def evaluation_config(self, tmp_path: Path) -> Config:
        """Provide a config for evaluation testing."""
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
            submission_path=tmp_path / "submission.csv",
            model_path=tmp_path / "model.pt",
        )

        return Config(data=data_config, model=model_config, output=output_config)

    @pytest.fixture
    def trained_model_and_metadata(self, evaluation_config: Config, device: torch.device):
        """Train a model and return it with metadata."""
        from mlp_binary_classification.data_loader import load_data
        from mlp_binary_classification.preprocessing import preprocess_titanic
        from mlp_binary_classification.train import train_model

        # First train to create checkpoint
        train_model(evaluation_config)

        # Load checkpoint
        checkpoint_path = evaluation_config.output.checkpoint_dir / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        input_size = checkpoint["input_size"]
        hidden_sizes = checkpoint["hidden_sizes"]
        dropout = checkpoint["dropout"]

        model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, dropout=dropout)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        # Get train metadata
        train_df, _ = load_data(evaluation_config.data)
        X_train, train_metadata = preprocess_titanic(train_df, evaluation_config.model, is_test=False)

        return model, train_metadata

    def test_evaluate_on_test_returns_tuple(
        self,
        evaluation_config: Config,
        trained_model_and_metadata: tuple,
    ) -> None:
        """Test that evaluate_on_test returns a tuple of (predictions, probabilities)."""
        model, train_metadata = trained_model_and_metadata
        _, test_df = evaluation_config.data.test_path, None  # Will load via function

        from mlp_binary_classification.data_loader import load_data

        _, test_df = load_data(evaluation_config.data)

        predictions, proba = evaluate_on_test(model, test_df, train_metadata, evaluation_config.model)

        assert isinstance(predictions, np.ndarray)
        assert isinstance(proba, np.ndarray)

    def test_evaluate_on_test_predictions_binary(
        self,
        evaluation_config: Config,
        trained_model_and_metadata: tuple,
    ) -> None:
        """Test that predictions are binary (0 or 1)."""
        model, train_metadata = trained_model_and_metadata

        from mlp_binary_classification.data_loader import load_data

        _, test_df = load_data(evaluation_config.data)

        predictions, _ = evaluate_on_test(model, test_df, train_metadata, evaluation_config.model)

        unique_vals = set(np.unique(predictions))
        assert unique_vals.issubset({0, 1})

    def test_evaluate_on_test_probabilities_range(
        self,
        evaluation_config: Config,
        trained_model_and_metadata: tuple,
    ) -> None:
        """Test that probabilities are in [0, 1]."""
        model, train_metadata = trained_model_and_metadata

        from mlp_binary_classification.data_loader import load_data

        _, test_df = load_data(evaluation_config.data)

        _, proba = evaluate_on_test(model, test_df, train_metadata, evaluation_config.model)

        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_evaluate_on_test_output_length(
        self,
        evaluation_config: Config,
        trained_model_and_metadata: tuple,
    ) -> None:
        """Test that output arrays have correct length."""
        model, train_metadata = trained_model_and_metadata

        from mlp_binary_classification.data_loader import load_data

        _, test_df = load_data(evaluation_config.data)

        predictions, proba = evaluate_on_test(model, test_df, train_metadata, evaluation_config.model)

        assert len(predictions) == len(test_df)
        assert len(proba) == len(test_df)


class TestEvaluateOnValidation:
    """Tests for evaluate_on_validation function."""

    @pytest.fixture
    def evaluation_config(self, tmp_path: Path) -> Config:
        """Provide a config for evaluation testing."""
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
            submission_path=tmp_path / "submission.csv",
            model_path=tmp_path / "model.pt",
        )

        return Config(data=data_config, model=model_config, output=output_config)

    def test_evaluate_on_validation_requires_checkpoint(
        self,
    ) -> None:
        """Test that evaluate_on_validation raises error when no checkpoint exists."""
        from pathlib import Path

        from mlp_binary_classification.config import Config, DataConfig, ModelConfig, OutputConfig

        tmp_path = Path("/tmp/test_no_checkpoint")
        tmp_path.mkdir(exist_ok=True)

        data_config = DataConfig(
            train_path=Path("tests/fixtures/sample_train.csv"),
            test_path=Path("tests/fixtures/sample_test.csv"),
        )
        model_config = ModelConfig()
        output_config = OutputConfig(
            checkpoint_dir=tmp_path / "nonexistent",
        )

        config = Config(data=data_config, model=model_config, output=output_config)

        with pytest.raises(FileNotFoundError):
            evaluate_on_validation(config)

    def test_evaluate_on_validation_metrics_range(
        self,
        evaluation_config: Config,
    ) -> None:
        """Test that all metrics are in valid ranges."""
        from mlp_binary_classification.train import train_model

        train_model(evaluation_config)

        metrics = evaluate_on_validation(evaluation_config)

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_evaluate_on_validation_confusion_matrix_shape(
        self,
        evaluation_config: Config,
    ) -> None:
        """Test that confusion matrix is 2x2."""
        from mlp_binary_classification.train import train_model

        train_model(evaluation_config)

        metrics = evaluate_on_validation(evaluation_config)

        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        for row in cm:
            assert len(row) == 2
