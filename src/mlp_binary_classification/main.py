# SPDX-FileCopyrightText: 2026 Maxim Selin (Mournweiss) <info@mournweiss.ru>
#
# SPDX-License-Identifier: Apache-2.0

"""Main entry point for the Titanic MLP classification project."""

import argparse

from .config import Config
from .evaluate import evaluate_on_validation
from .predict import generate_submission
from .train import train_model


def main() -> None:
    """Run the full pipeline: train, evaluate, predict."""
    parser = argparse.ArgumentParser(description="Titanic MLP Binary Classification")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "evaluate", "predict", "all"],
        help="Execution mode: train, evaluate, predict, or all",
    )
    args = parser.parse_args()

    config = Config()

    if args.mode in ("train", "all"):
        print("=== Training ===")
        history = train_model(config)
        print(f"Training complete. Final val loss: {history['val_loss_history'][-1]:.4f}")

    if args.mode in ("evaluate", "all"):
        print("\n=== Evaluation ===")
        metrics = evaluate_on_validation(config)

    if args.mode in ("predict", "all"):
        print("\n=== Prediction ===")
        submission = generate_submission(config)


if __name__ == "__main__":
    main()
