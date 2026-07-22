"""Data loading utilities for the Titanic dataset."""

from pathlib import Path

import pandas as pd

from .config import DataConfig


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded DataFrame.
    """
    return pd.read_csv(path)


def load_data(config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test datasets.

    Args:
        config: Data configuration with paths.

    Returns:
        Tuple of (train_df, test_df).
    """
    train_df = load_csv(config.train_path)
    test_df = load_csv(config.test_path)
    return train_df, test_df


def split_train_val(
    train_df: pd.DataFrame,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data into train and validation subsets.

    Args:
        train_df: Full training DataFrame.
        val_ratio: Fraction of data to use for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_split, val_split) DataFrames.
    """
    train_split = train_df.sample(frac=1 - val_ratio, random_state=random_state)
    val_split = train_df.drop(train_split.index)
    return train_split, val_split
