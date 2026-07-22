"""Training pipeline for the Titanic MLP model."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import Config, ModelConfig
from .data_loader import load_data, split_train_val
from .model import MLP
from .preprocessing import preprocess_titanic


def get_device() -> torch.device:
    """Get the best available computation device.

    Returns:
        torch.device: 'cuda', 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Args:
        X_train: Training features array.
        y_train: Training labels array.
        X_val: Validation features array.
        y_val: Validation labels array.
        batch_size: Batch size for DataLoader.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_model(config: Config) -> dict:
    """Train the MLP model on Titanic data.

    Args:
        config: Master configuration.

    Returns:
        Dictionary with training history and metadata.
    """
    model_config = config.model
    device = get_device()

    # Set seeds for reproducibility
    torch.manual_seed(model_config.seed)
    np.random.seed(model_config.seed)

    # Load data
    train_df, _ = load_data(config.data)
    train_split, val_split = split_train_val(
        train_df,
        val_ratio=config.data.val_split,
        random_state=config.data.random_state,
    )

    # Preprocess
    X_train, train_metadata = preprocess_titanic(train_split, model_config, is_test=False)
    y_train = train_split["Survived"].values.reshape(-1, 1)

    X_val, _ = preprocess_titanic(val_split, model_config, is_test=False)
    y_val = val_split["Survived"].values.reshape(-1, 1)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=model_config.batch_size)

    # Initialize model
    input_size = X_train.shape[1]
    model = MLP(
        input_size=input_size,
        hidden_sizes=model_config.hidden_sizes,
        dropout=model_config.dropout,
    )
    model._init_weights()
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_config.epochs)

    # Training loop
    loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_acc_history: list[float] = []

    best_val_loss = float("inf")

    for epoch in range(model_config.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.gradient_clip)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        scheduler.step()

        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            Path(config.output.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "input_size": input_size,
                    "hidden_sizes": model_config.hidden_sizes,
                    "dropout": model_config.dropout,
                },
                config.output.checkpoint_dir / "best_model.pt",
            )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{model_config.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

    history = {
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
        "val_acc_history": val_acc_history,
        "train_metadata": train_metadata,
        "input_size": input_size,
    }

    return history


def evaluate_model(
    model: MLP,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on validation data.

    Args:
        model: Trained MLP model.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Computation device.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X).squeeze(-1)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy
