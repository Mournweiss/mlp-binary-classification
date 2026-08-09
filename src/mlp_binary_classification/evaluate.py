# SPDX-FileCopyrightText: 2026 Maxim Selin (Mournweiss) <info@mournweiss.ru>
#
# SPDX-License-Identifier: Apache-2.0

"""Model evaluation utilities."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import Config, ModelConfig
from .data_loader import load_data, split_train_val
from .model import MLP
from .preprocessing import preprocess_test, preprocess_titanic
from .train import get_device


def evaluate_on_validation(config: Config) -> dict:
    """Evaluate the best model on validation data.

    Args:
        config: Master configuration.

    Returns:
        Dictionary with evaluation metrics.
    """
    model_config = config.model
    device = get_device()

    # Load and preprocess data
    train_df, _ = load_data(config.data)
    train_split, val_split = split_train_val(
        train_df,
        val_ratio=config.data.val_split,
        random_state=config.data.random_state,
    )

    X_train, train_metadata = preprocess_titanic(train_split, model_config, is_test=False)
    y_train = train_split["Survived"].values

    X_val, _ = preprocess_titanic(val_split, model_config, is_test=False)
    y_val = val_split["Survived"].values

    # Load best model
    checkpoint_path = config.output.checkpoint_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    input_size = checkpoint["input_size"]
    hidden_sizes = checkpoint["hidden_sizes"]
    dropout = checkpoint["dropout"]

    model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Predict
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    proba = model.predict_proba(X_val_tensor).cpu().numpy().flatten()
    predictions = (proba >= 0.5).astype(int)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(y_val, predictions, zero_division=0),
        "recall": recall_score(y_val, predictions, zero_division=0),
        "f1": f1_score(y_val, predictions, zero_division=0),
    }

    # ROC-AUC if possible
    try:
        metrics["roc_auc"] = roc_auc_score(y_val, proba)
    except ValueError:
        metrics["roc_auc"] = float("nan")

    metrics["confusion_matrix"] = confusion_matrix(y_val, predictions).tolist()
    metrics["classification_report"] = classification_report(y_val, predictions, zero_division=0)

    print("=== Validation Metrics ===")
    for key, value in metrics.items():
        if key != "confusion_matrix" and key != "classification_report":
            print(f"  {key}: {value:.4f}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"\nClassification Report:\n{metrics['classification_report']}")

    return metrics


def evaluate_on_test(
    model: MLP,
    test_df: pd.DataFrame,
    train_metadata: dict,
    model_config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate model on test data and return predictions + probabilities.

    Args:
        model: Trained MLP model.
        test_df: Test DataFrame.
        train_metadata: Metadata from training preprocessing.
        model_config: Model configuration.

    Returns:
        Tuple of (predictions, probabilities).
    """
    device = get_device()
    model.eval()

    X_test = preprocess_test(test_df, train_metadata, model_config)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        proba = model.predict_proba(X_test_tensor).cpu().numpy().flatten()
        predictions = (proba >= 0.5).astype(int)

    return predictions, proba
