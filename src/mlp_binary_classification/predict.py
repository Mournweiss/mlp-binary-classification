# SPDX-FileCopyrightText: 2026 Maxim Selin (Mournweiss) <info@mournweiss.ru>
#
# SPDX-License-Identifier: Apache-2.0

"""Prediction and submission generation."""

import pandas as pd
import torch

from .config import Config
from .data_loader import load_data
from .model import MLP
from .preprocessing import preprocess_titanic
from .train import get_device


def generate_submission(config: Config) -> pd.DataFrame:
    """Generate Kaggle submission CSV using the best model.

    Args:
        config: Master configuration.

    Returns:
        DataFrame with PassengerId and Survived predictions.
    """
    model_config = config.model
    device = get_device()

    # Load test data
    _, test_df = load_data(config.data)

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

    # Preprocess test data
    X_test, train_metadata = preprocess_titanic(test_df, model_config, is_test=True)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        proba = model.predict_proba(X_test_tensor).cpu().numpy().flatten()
        predictions = (proba >= 0.5).astype(int)

    # Create submission
    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Survived": predictions,
        }
    )

    submission.to_csv(config.output.submission_path, index=False)
    print(f"Submission saved to {config.output.submission_path}")
    print(f"Total predictions: {len(submission)}")
    print(f"Survived: {predictions.sum()} / {len(predictions)}")

    return submission
