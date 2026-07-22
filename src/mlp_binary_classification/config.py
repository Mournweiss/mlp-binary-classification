"""Configuration for the Titanic MLP classification project."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data paths and settings."""

    train_path: Path = field(default_factory=lambda: Path("data/train.csv"))
    test_path: Path = field(default_factory=lambda: Path("data/test.csv"))
    val_split: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    """MLP model hyperparameters."""

    input_size: int = 9
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 32, 16])
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 32
    gradient_clip: float = 1.0
    seed: int = 42


@dataclass
class OutputConfig:
    """Output paths and settings."""

    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    submission_path: Path = field(default_factory=lambda: Path("submission.csv"))
    model_path: Path = field(default_factory=lambda: Path("model.pt"))


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
