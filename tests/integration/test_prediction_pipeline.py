# SPDX-FileCopyrightText: 2026 Maxim Selin (Mournweiss) <info@mournweiss.ru>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the prediction pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mlp_binary_classification.config import Config, DataConfig, ModelConfig, OutputConfig
from mlp_binary_classification.data_loader import load_data
from mlp_binary_classification.model import MLP
from mlp_binary_classification.predict import generate_submission
from mlp_binary_classification.preprocessing import preprocess_titanic
from mlp_binary_classification.train import train_model


@pytest.fixture
def prediction_config(tmp_path: Path) -> Config:
    """Provide a config for prediction testing."""
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


@pytest.mark.integration
def test_generate_submission_creates_file(prediction_config: Config) -> None:
    """Test that generate_submission creates a CSV file."""
    train_model(prediction_config)

    submission = generate_submission(prediction_config)
    assert prediction_config.output.submission_path.exists()


@pytest.mark.integration
def test_submission_format(prediction_config: Config) -> None:
    """Test that submission CSV has correct format."""
    train_model(prediction_config)

    submission = generate_submission(prediction_config)

    # Check columns
    assert list(submission.columns) == ["PassengerId", "Survived"]

    # Check types
    assert submission["PassengerId"].dtype in (np.int64, np.int32, np.intp)
    assert submission["Survived"].dtype in (np.int64, np.int32, np.int8)

    # Check that it can be read back
    saved = pd.read_csv(prediction_config.output.submission_path)
    assert list(saved.columns) == ["PassengerId", "Survived"]


@pytest.mark.integration
def test_predict_with_trained_model(prediction_config: Config) -> None:
    """Test prediction with a trained model produces valid output."""
    train_model(prediction_config)

    submission = generate_submission(prediction_config)

    # Load test data for comparison
    _, test_df = load_data(prediction_config.data)

    # Check row count matches
    assert len(submission) == len(test_df)

    # Check PassengerIds match
    assert set(submission["PassengerId"]) == set(test_df["PassengerId"])

    # Check Survived values are binary
    assert set(submission["Survived"].unique()).issubset({0, 1})

    # Check no NaN values
    assert not submission.isnull().any().any()


@pytest.mark.integration
def test_full_training_to_prediction_pipeline(prediction_config: Config) -> None:
    """Test the full pipeline: train -> evaluate -> predict."""
    from mlp_binary_classification.evaluate import evaluate_on_validation

    # Train
    history = train_model(prediction_config)
    assert len(history["loss_history"]) == prediction_config.model.epochs

    # Evaluate
    metrics = evaluate_on_validation(prediction_config)
    assert "accuracy" in metrics
    assert "f1" in metrics

    # Predict
    submission = generate_submission(prediction_config)
    assert len(submission) > 0
    assert "Survived" in submission.columns


@pytest.mark.integration
def test_checkpoint_loadable_for_prediction(prediction_config: Config) -> None:
    """Test that checkpoint can be loaded for prediction."""
    train_model(prediction_config)

    checkpoint_path = prediction_config.output.checkpoint_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    # Verify checkpoint has all required keys
    required_keys = ["model_state_dict", "optimizer_state_dict", "val_loss", "input_size", "hidden_sizes", "dropout"]
    for key in required_keys:
        assert key in checkpoint

    # Verify model can be reconstructed
    input_size = checkpoint["input_size"]
    hidden_sizes = checkpoint["hidden_sizes"]
    dropout = checkpoint["dropout"]

    model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    assert model is not None
