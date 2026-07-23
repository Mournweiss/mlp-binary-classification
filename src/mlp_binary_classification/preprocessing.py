"""Preprocessing pipeline for Titanic dataset."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig


def extract_title(name: str) -> str:
    """Extract title from passenger name.

    Args:
        name: Full passenger name string.

    Returns:
        Extracted title (e.g., 'Mr', 'Miss', 'Mrs', 'Master', 'Other').
    """
    if "," not in name:
        return "Other"
    parts = name.split(",")
    if len(parts) < 2:
        return "Other"
    title_part = parts[1].strip()
    if "." not in title_part:
        return "Other"
    title = title_part.split(".")[0].strip()
    return title


def extract_deck(cabin: str) -> str:
    """Extract deck (cabin letter) from cabin number.

    Args:
        cabin: Cabin string (e.g., 'C85', 'B57').

    Returns:
        Deck letter or 'Unknown'.
    """
    if pd.isna(cabin) or not isinstance(cabin, str) or len(cabin) == 0:
        return "Unknown"
    return cabin[0].upper()


def preprocess_titanic(
    df: pd.DataFrame,
    config: ModelConfig,
    is_test: bool = False,
) -> tuple[np.ndarray, dict]:
    """Preprocess Titanic DataFrame into numerical features.

    Args:
        df: Input DataFrame (train or test).
        config: Model configuration.
        is_test: Whether this is the test set (no 'Survived' column).

    Returns:
        Tuple of (features_array, metadata_dict).
        metadata_dict contains fitted scalers/encoders for later use.
    """
    df = df.copy()

    # Extract new features
    df["Title"] = df["Name"].apply(extract_title)
    df["Deck"] = df["Cabin"].apply(extract_deck)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Fill missing Embarked with mode (for train)
    if not is_test and "Embarked" in df.columns:
        mode_embarked = df["Embarked"].mode()
        if len(mode_embarked) > 0:
            df["Embarked"] = df["Embarked"].fillna(mode_embarked.iloc[0])

    # Fill missing Fare with median (for test)
    if is_test and "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Fill missing Age with median
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Fill missing Deck with 'Unknown'
    df["Deck"] = df["Deck"].fillna("Unknown")

    # Encode Sex
    df["Sex_encoded"] = (df["Sex"] == "male").astype(int)

    # Encode Title
    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Other": 4}
    df["Title_encoded"] = df["Title"].map(title_mapping).fillna(4).astype(int)

    # One-hot encode Embarked
    if "Embarked" in df.columns:
        embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
        for col in ["Embarked_C", "Embarked_Q", "Embarked_S"]:
            if col not in embarked_dummies.columns:
                embarked_dummies[col] = 0
        df = pd.concat([df, embarked_dummies], axis=1)

    # One-hot encode Deck (top decks only to reduce dimensionality)
    top_decks = ["A", "B", "C", "D", "E", "F", "G", "T", "Unknown"]
    deck_dummies = pd.get_dummies(df["Deck"], prefix="Deck")
    for deck in top_decks:
        deck_col = f"Deck_{deck}"
        if deck_col not in deck_dummies.columns:
            deck_dummies[deck_col] = 0
    deck_cols = [c for c in deck_dummies.columns if c in top_decks or c in deck_dummies.columns]
    df = pd.concat([df, deck_dummies[deck_cols]], axis=1)

    # Select final features
    feature_cols = [
        "Pclass",
        "Sex_encoded",
        "Age",
        "Fare",
        "SibSp",
        "Parch",
        "Title_encoded",
        "FamilySize",
        "IsAlone",
    ]

    # Add Embarked one-hot columns
    for col in ["Embarked_C", "Embarked_Q", "Embarked_S"]:
        if col in df.columns:
            feature_cols.append(col)

    # Add Deck one-hot columns
    for deck in top_decks:
        deck_col = f"Deck_{deck}"
        if deck_col in df.columns:
            feature_cols.append(deck_col)

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    features = df[feature_cols].values.astype(np.float32)

    # Normalize numerical features
    scaler = StandardScaler()
    # Normalize first 9 features (before one-hot encoded columns)
    num_original = 9
    if features.shape[1] >= num_original:
        features[:, :num_original] = scaler.fit_transform(features[:, :num_original])

    metadata = {
        "feature_names": feature_cols,
        "scaler": scaler,
        "num_original_features": num_original,
    }

    return features, metadata


def preprocess_test(
    test_df: pd.DataFrame,
    train_metadata: dict,
    config: ModelConfig,
) -> np.ndarray:
    """Preprocess test set using metadata fitted on training data.

    Args:
        test_df: Test DataFrame.
        train_metadata: Metadata from training preprocessing.
        config: Model configuration.

    Returns:
        Preprocessed feature array.
    """
    features, _ = preprocess_titanic(test_df, config, is_test=True)

    # Apply training scaler
    scaler = train_metadata["scaler"]
    num_original = train_metadata["num_original_features"]
    features[:, :num_original] = scaler.transform(features[:, :num_original])

    return features
