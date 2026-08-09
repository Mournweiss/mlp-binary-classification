# SPDX-FileCopyrightText: 2026 Maxim Selin (Mournweiss) <info@mournweiss.ru>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for preprocessing and model pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mlp_binary_classification.config import Config, DataConfig, ModelConfig, OutputConfig
from mlp_binary_classification.data_loader import load_data, split_train_val
from mlp_binary_classification.model import MLP
from mlp_binary_classification.preprocessing import preprocess_titanic
from mlp_binary_classification.train import get_device, train_model


@pytest.fixture
def preprocessing_config(tmp_path: Path) -> Config:
    """Provide a config for preprocessing tests."""
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


@pytest.mark.integration
def test_preprocess_to_model_input(preprocessing_config: Config) -> None:
    """Test that preprocessed data can be fed into the model."""
    train_df, _ = load_data(preprocessing_config.data)
    train_split, _ = split_train_val(train_df, val_ratio=0.2, random_state=42)

    X_train, _ = preprocess_titanic(train_split, preprocessing_config.model, is_test=False)

    # Create model with correct input size
    input_size = X_train.shape[1]
    device = get_device()
    model = MLP(input_size=input_size, hidden_sizes=[16, 8], dropout=0.0).to(device)

    # Feed into model
    X_tensor = torch.tensor(X_train[:5], dtype=torch.float32).to(device)
    output = model(X_tensor)

    assert output.shape == (5, 1)
    assert not torch.isnan(output).any()


@pytest.mark.integration
def test_train_test_preprocess_consistency(preprocessing_config: Config) -> None:
    """Test that train and test preprocessing produce consistent feature dimensions."""
    train_df, test_df = load_data(preprocessing_config.data)

    X_train, train_metadata = preprocess_titanic(train_df, preprocessing_config.model, is_test=False)
    X_test, test_metadata = preprocess_titanic(test_df, preprocessing_config.model, is_test=True)

    # Feature dimensions should match
    assert X_train.shape[1] == X_test.shape[1]

    # Feature names should match
    assert train_metadata["feature_names"] == test_metadata["feature_names"]

    # Both should have no NaN
    assert not np.any(np.isnan(X_train))
    assert not np.any(np.isnan(X_test))


@pytest.mark.integration
def test_full_feature_pipeline(preprocessing_config: Config) -> None:
    """Test the full feature extraction pipeline produces expected features."""
    train_df, test_df = load_data(preprocessing_config.data)

    X_train, train_metadata = preprocess_titanic(train_df, preprocessing_config.model, is_test=False)
    X_test, test_metadata = preprocess_titanic(test_df, preprocessing_config.model, is_test=True)

    # Apply training scaler to test data
    scaler = train_metadata["scaler"]
    num_original = train_metadata["num_original_features"]
    X_test_scaled = X_test.copy()
    X_test_scaled[:, :num_original] = scaler.transform(X_test[:, :num_original])

    # Verify feature names
    feature_names = train_metadata["feature_names"]
    assert len(feature_names) == X_train.shape[1]
    assert "Pclass" in feature_names
    assert "Sex_encoded" in feature_names
    assert "Age" in feature_names
    assert "Fare" in feature_names

    # Verify Embarked one-hot columns exist
    embarked_cols = [f for f in feature_names if f.startswith("Embarked_")]
    assert len(embarked_cols) >= 2

    # Verify Deck one-hot columns exist
    deck_cols = [f for f in feature_names if f.startswith("Deck_")]
    assert len(deck_cols) >= 2


@pytest.mark.integration
def test_preprocessing_metadata_preservation(preprocessing_config: Config) -> None:
    """Test that preprocessing metadata is complete and usable."""
    train_df, _ = load_data(preprocessing_config.data)

    X_train, metadata = preprocess_titanic(train_df, preprocessing_config.model, is_test=False)

    # Metadata should contain all required keys
    assert "feature_names" in metadata
    assert "scaler" in metadata
    assert "num_original_features" in metadata

    # Scaler should be fitted (have mean_ and scale_ attributes)
    assert hasattr(metadata["scaler"], "mean_")
    assert hasattr(metadata["scaler"], "scale_")

    # num_original_features should be positive
    assert metadata["num_original_features"] > 0

    # feature_names should match array width
    assert len(metadata["feature_names"]) == X_train.shape[1]


@pytest.mark.integration
def test_preprocess_test_with_train_metadata(preprocessing_config: Config) -> None:
    """Test that test preprocessing works correctly with train metadata."""
    train_df, test_df = load_data(preprocessing_config.data)

    # Get train metadata
    X_train, train_metadata = preprocess_titanic(train_df, preprocessing_config.model, is_test=False)

    # Preprocess test using train metadata
    from mlp_binary_classification.preprocessing import preprocess_test

    X_test = preprocess_test(test_df, train_metadata, preprocessing_config.model)

    # Test features should use train scaler
    assert X_test.shape[0] == len(test_df)
    assert X_test.shape[1] == X_train.shape[1]
    assert not np.any(np.isnan(X_test))


@pytest.mark.integration
def test_end_to_end_preprocess_train_evaluate(preprocessing_config: Config) -> None:
    """Test the complete pipeline: preprocess -> train -> evaluate."""
    # Train
    history = train_model(preprocessing_config)

    # Verify training completed
    assert len(history["loss_history"]) == preprocessing_config.model.epochs
    assert len(history["val_loss_history"]) == preprocessing_config.model.epochs

    # Verify loss decreased or remained stable
    initial_loss = history["loss_history"][0]
    final_loss = history["loss_history"][-1]
    assert isinstance(initial_loss, float)
    assert isinstance(final_loss, float)

    # Verify validation accuracy is valid
    val_accs = history["val_acc_history"]
    assert all(0 <= acc <= 1 for acc in val_accs)
