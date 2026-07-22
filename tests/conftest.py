"""Shared fixtures for all tests."""

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture(scope="session")
def sample_train_df() -> pd.DataFrame:
    """Provide a small sample training DataFrame for testing."""
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Survived": [0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 3, 2],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
                "Futrelle, Mrs. Jacques Heath",
                "Allen, Mr. William Henry",
                "Moran, Mr. James",
                "McCarthy, Mr. Timothy J",
                "Palsson, Master. Gosta Leonard",
                "Johnson, Mrs. Oscar W",
                "Nasser, Mrs. Nicholas",
            ],
            "Sex": ["male", "female", "female", "female", "male", "male", "male", "male", "female", "female"],
            "Age": [22, 38, 26, 35, 35, None, 54, 2, 27, 14],
            "SibSp": [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
            "Parch": [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
            "Ticket": [
                "A/5",
                "PC 17599",
                "STON/O2",
                "113803",
                "373450",
                "330877",
                "17463",
                "349909",
                "347742",
                "237736",
            ],
            "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075, 11.1333, 30.0708],
            "Cabin": [None, "C85", None, "C123", None, None, "E46", None, None, None],
            "Embarked": ["S", "C", "S", "S", "S", "Q", "S", "S", "S", "C"],
        }
    )


@pytest.fixture(scope="session")
def sample_test_df() -> pd.DataFrame:
    """Provide a small sample test DataFrame for testing."""
    return pd.DataFrame(
        {
            "PassengerId": [11, 12, 13, 14, 15],
            "Pclass": [3, 1, 2, 3, 1],
            "Name": [
                "Test, Mr. John",
                "Doe, Mrs. Jane",
                "Smith, Mr. Bob",
                "Johnson, Miss. Alice",
                "Brown, Mr. Charles",
            ],
            "Sex": ["male", "female", "male", "female", "male"],
            "Age": [30, None, 25, 40, 50],
            "SibSp": [0, 1, 0, 0, 2],
            "Parch": [0, 0, 0, 1, 0],
            "Ticket": ["T1", "T2", "T3", "T4", "T5"],
            "Fare": [10.0, 50.0, 20.0, None, 15.0],
            "Cabin": [None, "D10", None, "B20", None],
            "Embarked": ["S", "C", "Q", "S", "C"],
        }
    )


@pytest.fixture(scope="session")
def sample_features() -> np.ndarray:
    """Provide sample normalized feature array for model testing."""
    return np.array(
        [
            [0.5, 1.0, 0.0, 0.3, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [-0.5, 0.0, -0.5, -0.2, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.3, 0.5, 2.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture(scope="session")
def sample_labels() -> np.ndarray:
    """Provide sample binary labels for testing."""
    return np.array([[1], [0], [1]], dtype=np.float32)


@pytest.fixture(scope="function")
def device() -> torch.device:
    """Provide computation device for model tests."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="function")
def dummy_model(device: torch.device) -> torch.nn.Module:
    """Provide a small MLP model for testing."""
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(19, 32),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(32, 1),
    )
    return model.to(device)
