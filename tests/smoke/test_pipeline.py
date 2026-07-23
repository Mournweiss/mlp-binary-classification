"""Smoke tests for quick sanity checks of the Titanic MLP pipeline."""

import pytest
import torch


@pytest.mark.smoke
def test_import_all_modules() -> None:
    """Verify all modules can be imported without errors."""
    from mlp_binary_classification import config, data_loader, evaluate, model, predict, preprocessing, train

    assert config is not None
    assert data_loader is not None
    assert model is not None
    assert preprocessing is not None
    assert train is not None
    assert evaluate is not None
    assert predict is not None


@pytest.mark.smoke
def test_config_instantiation() -> None:
    """Verify Config can be instantiated with default values."""
    from mlp_binary_classification.config import Config

    config = Config()
    assert config.data is not None
    assert config.model is not None
    assert config.output is not None
    assert config.data.train_path.name == "train.csv"
    assert config.model.epochs == 50
    assert config.model.hidden_sizes == [64, 32, 16]


@pytest.mark.smoke
def test_model_creation_and_forward() -> None:
    """Verify MLP model can be created and performs a forward pass."""
    from mlp_binary_classification.model import MLP

    model = MLP(input_size=19, hidden_sizes=[32, 16], dropout=0.1)
    x = torch.randn(2, 19)
    output = model(x)
    assert output.shape == (2, 1)
    assert not torch.isnan(output).any()
