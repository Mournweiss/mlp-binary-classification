"""Integration tests for the evaluation pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mlp_binary_classification.config import Config, DataConfig, ModelConfig, OutputConfig
from mlp_binary_classification.data_loader import load_data, split_train_val
from mlp_binary_classification.evaluate import (
    evaluate_on_test,
    evaluate_on_validation,
    preprocess_test,
)
from mlp_binary_classification.model import MLP
from mlp_binary_classification.preprocessing import preprocess_titanic
from mlp_binary_classification.train import train_model


@pytest.fixture
def evaluation_config(tmp_path: Path) -> Config:
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
        hidden_sizes=[16, 8],
        epochs=5,
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
def trained_model_and_metadata(evaluation_config: Config, device: torch.device):
    """Train a model and return it with metadata."""
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


@pytest.mark.integration
def test_evaluate_on_validation(evaluation_config: Config) -> None:
    """Test evaluation on validation set produces valid metrics."""
    # First train a model
    from mlp_binary_classification.train import train_model

    train_model(evaluation_config)

    metrics = evaluate_on_validation(evaluation_config)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics

    # Accuracy should be in valid range
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1


@pytest.mark.integration
def test_metrics_computation(evaluation_config: Config) -> None:
    """Test that metrics are computed correctly."""
    from mlp_binary_classification.train import train_model

    train_model(evaluation_config)

    metrics = evaluate_on_validation(evaluation_config)

    # Confusion matrix should be a 2x2 list of lists
    cm = metrics["confusion_matrix"]
    assert len(cm) == 2
    for row in cm:
        assert len(row) == 2
        assert all(isinstance(v, int) for v in row)

    # Classification report should be a non-empty string
    assert isinstance(metrics["classification_report"], str)
    assert len(metrics["classification_report"]) > 0


@pytest.mark.integration
def test_roc_auc_computation(evaluation_config: Config) -> None:
    """Test ROC-AUC computation."""
    from mlp_binary_classification.train import train_model

    train_model(evaluation_config)

    metrics = evaluate_on_validation(evaluation_config)

    # ROC-AUC should be present (may be NaN if only one class)
    assert "roc_auc" in metrics


@pytest.mark.integration
def test_preprocess_test_function(evaluation_config: Config) -> None:
    """Test preprocess_test function."""
    _, test_df = load_data(evaluation_config.data)

    # Get train metadata first
    train_df, _ = load_data(evaluation_config.data)
    X_train, train_metadata = preprocess_titanic(train_df, evaluation_config.model, is_test=False)

    # Preprocess test
    X_test = preprocess_test(test_df, train_metadata, evaluation_config.model)

    assert isinstance(X_test, np.ndarray)
    assert X_test.dtype == np.float32
    assert X_test.shape[0] == len(test_df)
    assert not np.any(np.isnan(X_test))


@pytest.mark.integration
def test_evaluate_on_test_function(evaluation_config: Config, device: torch.device) -> None:
    """Test evaluate_on_test function."""
    from mlp_binary_classification.train import train_model

    train_model(evaluation_config)

    _, test_df = load_data(evaluation_config.data)
    train_df, _ = load_data(evaluation_config.data)
    _, train_metadata = preprocess_titanic(train_df, evaluation_config.model, is_test=False)

    checkpoint_path = evaluation_config.output.checkpoint_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    input_size = checkpoint["input_size"]
    hidden_sizes = checkpoint["hidden_sizes"]
    dropout = checkpoint["dropout"]

    model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    predictions, proba = evaluate_on_test(model, test_df, train_metadata, evaluation_config.model)

    assert isinstance(predictions, np.ndarray)
    assert isinstance(proba, np.ndarray)
    assert len(predictions) == len(test_df)
    assert len(proba) == len(test_df)
    # Predictions should be binary
    assert set(np.unique(predictions)).issubset({0, 1})
    # Probabilities should be in [0, 1]
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)
