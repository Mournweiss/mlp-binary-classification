# SPDX-FileCopyrightText: 2026 Maxim Selin (Mournweiss) <info@mournweiss.ru>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for configuration module."""

from pathlib import Path

from mlp_binary_classification.config import (
    Config,
    DataConfig,
    ModelConfig,
    OutputConfig,
)


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_paths(self) -> None:
        """Test default data paths are set correctly."""
        config = DataConfig()
        assert config.train_path.name == "train.csv"
        assert config.test_path.name == "test.csv"
        assert config.val_split == 0.2
        assert config.random_state == 42

    def test_custom_paths(self, tmp_path: Path) -> None:
        """Test custom data paths."""
        custom_train = tmp_path / "custom_train.csv"
        custom_train.touch()
        custom_test = tmp_path / "custom_test.csv"
        custom_test.touch()

        config = DataConfig(
            train_path=custom_train,
            test_path=custom_test,
            val_split=0.3,
            random_state=123,
        )
        assert config.train_path == custom_train
        assert config.test_path == custom_test
        assert config.val_split == 0.3
        assert config.random_state == 123


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default model hyperparameters."""
        config = ModelConfig()
        assert config.input_size == 9
        assert config.hidden_sizes == [64, 32, 16]
        assert config.dropout == 0.3
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 1e-4
        assert config.epochs == 50
        assert config.batch_size == 32
        assert config.gradient_clip == 1.0
        assert config.seed == 42

    def test_custom_hidden_sizes(self) -> None:
        """Test custom hidden layer sizes."""
        config = ModelConfig(hidden_sizes=[128, 64, 32, 16])
        assert config.hidden_sizes == [128, 64, 32, 16]


class TestOutputConfig:
    """Tests for OutputConfig dataclass."""

    def test_default_paths(self) -> None:
        """Test default output paths."""
        config = OutputConfig()
        assert config.checkpoint_dir.name == "checkpoints"
        assert config.submission_path.name == "submission.csv"
        assert config.model_path.name == "model.pt"


class TestConfig:
    """Tests for master Config dataclass."""

    def test_default_config(self) -> None:
        """Test default master configuration."""
        config = Config()
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.output, OutputConfig)

    def test_custom_config(self) -> None:
        """Test custom master configuration."""
        data_config = DataConfig(val_split=0.25, random_state=99)
        model_config = ModelConfig(epochs=100, batch_size=64)
        output_config = OutputConfig(submission_path=Path("my_submission.csv"))

        config = Config(data=data_config, model=model_config, output=output_config)
        assert config.data.val_split == 0.25
        assert config.data.random_state == 99
        assert config.model.epochs == 100
        assert config.model.batch_size == 64
        assert config.output.submission_path == Path("my_submission.csv")
