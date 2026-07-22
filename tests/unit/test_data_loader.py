"""Unit tests for data loader module."""

import pandas as pd
import pytest

from mlp_binary_classification.config import DataConfig
from mlp_binary_classification.data_loader import (
    load_csv,
    split_train_val,
)


class TestLoadCSV:
    """Tests for load_csv function."""

    def test_load_csv_returns_dataframe(self, tmp_path) -> None:
        """Test that load_csv returns a DataFrame."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")

        result = load_csv(csv_file)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_load_csv_empty_file(self, tmp_path) -> None:
        """Test loading an empty CSV file."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        with pytest.raises(pd.errors.EmptyDataError):
            load_csv(csv_file)

    def test_load_csv_single_column(self, tmp_path) -> None:
        """Test loading a single-column CSV."""
        csv_file = tmp_path / "single.csv"
        csv_file.write_text("col\n1\n2\n3\n")

        result = load_csv(csv_file)
        assert list(result.columns) == ["col"]
        assert len(result) == 3


class TestSplitTrainVal:
    """Tests for split_train_val function."""

    def test_split_preserves_rows(self, sample_train_df: pd.DataFrame) -> None:
        """Test that split preserves total row count."""
        train_split, val_split = split_train_val(sample_train_df)
        assert len(train_split) + len(val_split) == len(sample_train_df)

    def test_split_no_overlap(self, sample_train_df: pd.DataFrame) -> None:
        """Test that train and val sets have no overlapping indices."""
        train_split, val_split = split_train_val(sample_train_df)
        assert len(set(train_split.index) & set(val_split.index)) == 0

    def test_split_preserves_columns(self, sample_train_df: pd.DataFrame) -> None:
        """Test that split preserves all columns."""
        train_split, val_split = split_train_val(sample_train_df)
        assert list(train_split.columns) == list(sample_train_df.columns)
        assert list(val_split.columns) == list(sample_train_df.columns)

    def test_split_custom_ratio(self, sample_train_df: pd.DataFrame) -> None:
        """Test split with custom validation ratio."""
        train_split, val_split = split_train_val(sample_train_df, val_ratio=0.5, random_state=42)
        expected_train = int(len(sample_train_df) * 0.5)
        expected_val = len(sample_train_df) - expected_train
        assert len(train_split) == expected_train
        assert len(val_split) == expected_val

    def test_split_reproducibility(self, sample_train_df: pd.DataFrame) -> None:
        """Test that split is reproducible with same random_state."""
        train1, val1 = split_train_val(sample_train_df, random_state=42)
        train2, val2 = split_train_val(sample_train_df, random_state=42)
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)

    def test_split_zero_val_ratio(self, sample_train_df: pd.DataFrame) -> None:
        """Test split with zero validation ratio."""
        train_split, val_split = split_train_val(sample_train_df, val_ratio=0.0, random_state=42)
        assert len(train_split) == len(sample_train_df)
        assert len(val_split) == 0

    def test_split_all_val_ratio(self, sample_train_df: pd.DataFrame) -> None:
        """Test split with all validation ratio."""
        train_split, val_split = split_train_val(sample_train_df, val_ratio=1.0, random_state=42)
        assert len(train_split) == 0
        assert len(val_split) == len(sample_train_df)
