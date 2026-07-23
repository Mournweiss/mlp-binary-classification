"""Unit tests for the prediction module."""

from pathlib import Path

import pandas as pd
import pytest

from mlp_binary_classification.config import Config, DataConfig, ModelConfig, OutputConfig
from mlp_binary_classification.predict import generate_submission


class TestGenerateSubmission:
    """Tests for generate_submission function."""

    @pytest.fixture
    def prediction_config(self, tmp_path: Path) -> Config:
        """Provide a config for prediction testing."""
        sample_train = Path("tests/fixtures/sample_train.csv")
        sample_test = Path("tests/fixtures/sample_test.csv")

        data_config = DataConfig(
            train_path=sample_train,
            test_path=sample_test,
            val_split=0.2,
            random_state=42,
        )
        model_config = ModelConfig(
            hidden_sizes=[8, 4],
            epochs=3,
            batch_size=4,
            learning_rate=1e-2,
            seed=42,
        )
        output_config = OutputConfig(
            checkpoint_dir=tmp_path / "checkpoints",
            submission_path=tmp_path / "submission.csv",
            model_path=tmp_path / "model.pt",
        )

        return Config(data=data_config, model=model_config, output=output_config)

    def test_generate_submission_returns_dataframe(
        self,
        prediction_config: Config,
    ) -> None:
        """Test that generate_submission returns a pandas DataFrame."""
        from mlp_binary_classification.train import train_model

        train_model(prediction_config)

        submission = generate_submission(prediction_config)

        assert isinstance(submission, pd.DataFrame)

    def test_generate_submission_has_required_columns(
        self,
        prediction_config: Config,
    ) -> None:
        """Test that submission has PassengerId and Survived columns."""
        from mlp_binary_classification.train import train_model

        train_model(prediction_config)

        submission = generate_submission(prediction_config)

        assert "PassengerId" in submission.columns
        assert "Survived" in submission.columns

    def test_generate_submission_correct_row_count(
        self,
        prediction_config: Config,
    ) -> None:
        """Test that submission has correct number of rows."""
        from mlp_binary_classification.data_loader import load_data
        from mlp_binary_classification.train import train_model

        train_model(prediction_config)

        _, test_df = load_data(prediction_config.data)
        submission = generate_submission(prediction_config)

        assert len(submission) == len(test_df)

    def test_generate_submission_saved_to_csv(
        self,
        prediction_config: Config,
    ) -> None:
        """Test that submission is saved to CSV file."""
        from mlp_binary_classification.train import train_model

        train_model(prediction_config)

        submission = generate_submission(prediction_config)
        submission_path = prediction_config.output.submission_path

        assert submission_path.exists()

        # Verify the saved file can be read back
        saved_submission = pd.read_csv(submission_path)
        pd.testing.assert_frame_equal(saved_submission, submission)

    def test_generate_submission_predictions_binary(
        self,
        prediction_config: Config,
    ) -> None:
        """Test that Survived predictions are binary (0 or 1)."""
        from mlp_binary_classification.train import train_model

        train_model(prediction_config)

        submission = generate_submission(prediction_config)

        unique_vals = set(submission["Survived"].unique())
        assert unique_vals.issubset({0, 1})

    def test_generate_submission_passenger_ids_match(
        self,
        prediction_config: Config,
    ) -> None:
        """Test that PassengerIds match the test set."""
        from mlp_binary_classification.data_loader import load_data
        from mlp_binary_classification.train import train_model

        train_model(prediction_config)

        _, test_df = load_data(prediction_config.data)
        submission = generate_submission(prediction_config)

        pd.testing.assert_series_equal(
            submission["PassengerId"].sort_values().reset_index(drop=True),
            test_df["PassengerId"].sort_values().reset_index(drop=True),
        )

    def test_generate_submission_no_nan_values(
        self,
        prediction_config: Config,
    ) -> None:
        """Test that submission has no NaN values."""
        from mlp_binary_classification.train import train_model

        train_model(prediction_config)

        submission = generate_submission(prediction_config)

        assert not submission.isnull().any().any()
