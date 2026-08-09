# SPDX-FileCopyrightText: 2026 Maxim Selin (Mournweiss) <info@mournweiss.ru>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the main CLI entry point."""

import sys
from unittest.mock import patch

import pytest

from mlp_binary_classification.main import main


class TestMainCLI:
    """Tests for main CLI function."""

    @pytest.fixture
    def cli_config(self, tmp_path) -> dict:
        """Provide a minimal config for CLI tests."""
        from pathlib import Path

        from mlp_binary_classification.config import Config, DataConfig, ModelConfig, OutputConfig

        sample_train = Path("tests/fixtures/sample_train.csv")
        sample_test = Path("tests/fixtures/sample_test.csv")

        data_config = DataConfig(
            train_path=sample_train,
            test_path=sample_test,
        )
        model_config = ModelConfig(
            hidden_sizes=[8, 4],
            epochs=2,
            batch_size=4,
            seed=42,
        )
        output_config = OutputConfig(
            checkpoint_dir=tmp_path / "checkpoints",
            submission_path=tmp_path / "submission.csv",
            model_path=tmp_path / "model.pt",
        )

        return Config(data=data_config, model=model_config, output=output_config)

    def test_mode_train_runs_without_error(self, cli_config) -> None:
        """Test that --mode train runs without error."""
        with (
            patch("mlp_binary_classification.main.Config") as MockConfig,
            patch("mlp_binary_classification.main.train_model") as mock_train,
            patch("sys.stdout", new_callable=lambda: __import__("io").StringIO()),
            patch.object(__import__("sys"), "argv", ["main.py", "--mode", "train"]),
        ):
            MockConfig.return_value = cli_config
            mock_train.return_value = {"val_loss_history": [0.5, 0.3]}

            main()
            mock_train.assert_called_once()

    def test_mode_evaluate_runs_without_error(self, cli_config) -> None:
        """Test that --mode evaluate runs without error."""
        from mlp_binary_classification.train import train_model

        # First train to create checkpoint
        train_model(cli_config)

        with (
            patch("mlp_binary_classification.main.Config") as MockConfig,
            patch("mlp_binary_classification.main.evaluate_on_validation") as mock_eval,
            patch("sys.stdout", new_callable=lambda: __import__("io").StringIO()),
        ):
            MockConfig.return_value = cli_config
            mock_eval.return_value = {"accuracy": 0.8, "f1": 0.7}

            # Simulate --mode evaluate
            with patch.object(sys, "argv", ["main.py", "--mode", "evaluate"]):
                main()
            mock_eval.assert_called_once()

    def test_mode_predict_runs_without_error(self, cli_config) -> None:
        """Test that --mode predict runs without error."""
        from mlp_binary_classification.train import train_model

        # First train to create checkpoint
        train_model(cli_config)

        with (
            patch("mlp_binary_classification.main.Config") as MockConfig,
            patch("mlp_binary_classification.main.generate_submission") as mock_predict,
            patch("sys.stdout", new_callable=lambda: __import__("io").StringIO()),
        ):
            MockConfig.return_value = cli_config
            mock_predict.return_value = __import__("pandas").DataFrame({"PassengerId": [1, 2], "Survived": [0, 1]})

            # Simulate --mode predict
            with patch.object(sys, "argv", ["main.py", "--mode", "predict"]):
                main()
            mock_predict.assert_called_once()

    def test_mode_all_runs_all_stages(self, cli_config) -> None:
        """Test that --mode all runs train, evaluate, and predict."""
        from mlp_binary_classification.train import train_model

        # First train to create checkpoint
        train_model(cli_config)

        with (
            patch("mlp_binary_classification.main.Config") as MockConfig,
            patch("mlp_binary_classification.main.train_model") as mock_train,
            patch("mlp_binary_classification.main.evaluate_on_validation") as mock_eval,
            patch("mlp_binary_classification.main.generate_submission") as mock_predict,
            patch("sys.stdout", new_callable=lambda: __import__("io").StringIO()),
        ):
            MockConfig.return_value = cli_config
            mock_train.return_value = {"val_loss_history": [0.5]}
            mock_eval.return_value = {"accuracy": 0.8}
            mock_predict.return_value = __import__("pandas").DataFrame({"PassengerId": [1], "Survived": [0]})

            # Simulate --mode all
            with patch.object(sys, "argv", ["main.py", "--mode", "all"]):
                main()

            mock_train.assert_called_once()
            mock_eval.assert_called_once()
            mock_predict.assert_called_once()

    def test_default_mode_is_all(self, cli_config) -> None:
        """Test that default mode is 'all'."""
        with (
            patch("mlp_binary_classification.main.Config") as MockConfig,
            patch("mlp_binary_classification.main.train_model") as mock_train,
            patch("mlp_binary_classification.main.evaluate_on_validation") as mock_eval,
            patch("mlp_binary_classification.main.generate_submission") as mock_predict,
            patch("sys.stdout", new_callable=lambda: __import__("io").StringIO()),
        ):
            MockConfig.return_value = cli_config
            mock_train.return_value = {"val_loss_history": [0.5]}
            mock_eval.return_value = {"accuracy": 0.8}
            mock_predict.return_value = __import__("pandas").DataFrame({"PassengerId": [1], "Survived": [0]})

            # No --mode argument, should default to 'all'
            with patch.object(sys, "argv", ["main.py"]):
                main()

            mock_train.assert_called_once()
            mock_eval.assert_called_once()
            mock_predict.assert_called_once()

    def test_invalid_mode_raises_error(self) -> None:
        """Test that invalid mode raises SystemExit."""
        with patch.object(sys, "argv", ["main.py", "--mode", "invalid"]):
            with pytest.raises(SystemExit):
                main()
