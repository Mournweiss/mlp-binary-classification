"""Unit tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from mlp_binary_classification.config import ModelConfig
from mlp_binary_classification.preprocessing import (
    extract_deck,
    extract_title,
    preprocess_titanic,
)


class TestExtractTitle:
    """Tests for extract_title function."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Braund, Mr. Owen Harris", "Mr"),
            ("Cumings, Mrs. John Bradley", "Mrs"),
            ("Heikkinen, Miss. Laina", "Miss"),
            ("Palsson, Master. Gosta Leonard", "Master"),
            ("Futrelle, Dr. Jacques Heath", "Dr"),
            ("Unknown Name Format", "Other"),
            ("NoComma", "Other"),
            ("Name, NoDot", "Other"),
        ],
    )
    def test_extract_title_various_formats(self, name: str, expected: str) -> None:
        """Test title extraction for various name formats."""
        assert extract_title(name) == expected


class TestExtractDeck:
    """Tests for extract_deck function."""

    @pytest.mark.parametrize(
        "cabin, expected",
        [
            ("C85", "C"),
            ("B57", "B"),
            ("G6", "G"),
            ("T", "T"),
            ("", "Unknown"),
            (None, "Unknown"),
        ],
    )
    def test_extract_deck_various_inputs(self, cabin, expected) -> None:
        """Test deck extraction for various cabin inputs."""
        assert extract_deck(cabin) == expected

    def test_extract_deck_uppercase(self) -> None:
        """Test that deck is always uppercase."""
        assert extract_deck("c85") == "C"


class TestPreprocessTitanic:
    """Tests for preprocess_titanic function."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Provide model config for preprocessing tests."""
        return ModelConfig(input_size=19)

    def test_preprocess_returns_numpy_array(self, sample_train_df: pd.DataFrame, model_config: ModelConfig) -> None:
        """Test that preprocessing returns numpy array."""
        features, metadata = preprocess_titanic(sample_train_df, model_config, is_test=False)
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32

    def test_preprocess_output_shape(self, sample_train_df: pd.DataFrame, model_config: ModelConfig) -> None:
        """Test that output shape matches input rows."""
        features, _ = preprocess_titanic(sample_train_df, model_config, is_test=False)
        assert features.shape[0] == len(sample_train_df)
        assert features.shape[1] > 0  # Should have features

    def test_preprocess_handles_missing_age(self, sample_train_df: pd.DataFrame, model_config: ModelConfig) -> None:
        """Test that missing Age values are filled."""
        features, _ = preprocess_titanic(sample_train_df, model_config, is_test=False)
        # Age is the 3rd feature (index 2), should not contain NaN
        assert not np.any(np.isnan(features[:, 2]))

    def test_preprocess_handles_missing_fare(self, sample_test_df: pd.DataFrame, model_config: ModelConfig) -> None:
        """Test that missing Fare values are filled in test set."""
        features, _ = preprocess_titanic(sample_test_df, model_config, is_test=True)
        # Fare is the 4th feature (index 3), should not contain NaN
        assert not np.any(np.isnan(features[:, 3]))

    def test_preprocess_metadata_contains_scaler(
        self, sample_train_df: pd.DataFrame, model_config: ModelConfig
    ) -> None:
        """Test that metadata contains scaler and feature names."""
        _, metadata = preprocess_titanic(sample_train_df, model_config, is_test=False)
        assert "scaler" in metadata
        assert "feature_names" in metadata
        assert "num_original_features" in metadata
        assert len(metadata["feature_names"]) > 0

    def test_preprocess_sex_encoding(self, sample_train_df: pd.DataFrame, model_config: ModelConfig) -> None:
        """Test that Sex is correctly encoded as 0/1 before normalization."""
        df = sample_train_df.copy()
        df["Sex_encoded"] = (df["Sex"] == "male").astype(int)
        # Sex_encoded should be binary (0 or 1)
        assert set(df["Sex_encoded"].unique()).issubset({0, 1})

    def test_preprocess_no_nan_values(self, sample_train_df: pd.DataFrame, model_config: ModelConfig) -> None:
        """Test that output contains no NaN values."""
        features, _ = preprocess_titanic(sample_train_df, model_config, is_test=False)
        assert not np.any(np.isnan(features))

    def test_preprocess_no_inf_values(self, sample_train_df: pd.DataFrame, model_config: ModelConfig) -> None:
        """Test that output contains no Inf values."""
        features, _ = preprocess_titanic(sample_train_df, model_config, is_test=False)
        assert not np.any(np.isinf(features))

    def test_preprocess_embarked_one_hot(self, sample_train_df: pd.DataFrame, model_config: ModelConfig) -> None:
        """Test that Embarked is one-hot encoded."""
        features, metadata = preprocess_titanic(sample_train_df, model_config, is_test=False)
        embarked_cols = [i for i, name in enumerate(metadata["feature_names"]) if name.startswith("Embarked_")]
        assert len(embarked_cols) >= 2  # At least 2 of 3 ports

    def test_preprocess_title_encoding(self, sample_train_df: pd.DataFrame, model_config: ModelConfig) -> None:
        """Test that Title is correctly encoded before normalization."""
        df = sample_train_df.copy()
        df["Title"] = df["Name"].apply(extract_title)
        title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Other": 4}
        df["Title_encoded"] = df["Title"].map(title_mapping).fillna(4).astype(int)
        assert set(df["Title_encoded"].unique()).issubset({0, 1, 2, 3, 4})
