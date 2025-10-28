"""
Deep Learning Models: DeepLOB and Transformer
For price direction prediction from order book snapshots
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict
import pickle

from config import *
from utils.io_utils import read_parquet

# logging.basicConfig(**LOGGING_CONFIG)
configure_logging()
logger = logging.getLogger(__name__)


class OrderBookDataset(Dataset):
    """Dataset for order book sequences"""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 100
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.labels[idx + self.sequence_length]
        return x, y


class DeepLOB(nn.Module):
    """
    DeepLOB: Deep Convolutional Neural Network for Limit Order Book

    Based on: Zhang et al. (2019) "DeepLOB: Deep Convolutional Neural Networks
    for Limit Order Books"
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        hidden_dims: list = [64, 128, 128],
        kernel_sizes: list = [3, 3, 3],
        dropout: float = 0.2
    ):
        super(DeepLOB, self).__init__()

        self.n_features = n_features
        self.n_classes = n_classes

        # Convolutional layers
        self.conv1 = nn.Conv1d(n_features, hidden_dims[0], kernel_size=kernel_sizes[0])
        self.conv2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=kernel_sizes[1])
        self.conv3 = nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size=kernel_sizes[2])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Calculate output size after convolutions
        # This will be sequence_length - sum(kernel_sizes) + len(kernel_sizes)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dims[2], 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # x shape: (batch, sequence, features)
        # Conv1d expects: (batch, features, sequence)
        x = x.transpose(1, 2)

        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.dropout(x)

        x = self.relu(self.conv2(x))
        x = self.dropout(x)

        x = self.relu(self.conv3(x))
        x = self.dropout(x)

        # Pooling
        x = self.pool(x).squeeze(-1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)

        return x


class TransformerLOB(nn.Module):
    """
    Transformer model for order book prediction

    Uses multi-head attention to capture temporal dependencies
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        n_classes: int = 3
    ):
        super(TransformerLOB, self).__init__()

        self.n_features = n_features
        self.d_model = d_model
        self.n_classes = n_classes

        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )

        # Output layers
        self.fc_out = nn.Linear(d_model, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, sequence, features)

        # Project to d_model dimensions
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Take last timestep output
        x = x[:, -1, :]

        # Dropout and classification
        x = self.dropout(x)
        x = self.fc_out(x)

        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch, sequence, d_model)
        x = x + self.pe[:x.size(1), 0, :].unsqueeze(0)
        return self.dropout(x)


class ModelTrainer:
    """Trainer for deep learning models"""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None

    def setup_optimizer(
        self,
        learning_rate: float = 0.001,
        optimizer_type: str = "adam"
    ):
        """Setup optimizer and scheduler"""
        if optimizer_type == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=learning_rate, momentum=0.9
            )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10
    ) -> Dict[str, list]:
        """
        Full training loop with early stopping

        Returns:
            Dictionary with training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_accuracy = self.validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.2f}%"
            )

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return history

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions

        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                output = self.model(data)

                # Apply softmax to get probabilities
                probs = torch.softmax(output, dim=1)

                _, preds = torch.max(output, 1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)

    def save_model(self, filepath: Path):
        """Save model weights"""
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Path):
        """Load model weights"""
        self.model.load_state_dict(torch.load(filepath))
        logger.info(f"Model loaded from {filepath}")


def prepare_data(
    df: pd.DataFrame,
    feature_cols: list,
    label_col: str = "label",
    sequence_length: int = 100,
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare data loaders for training

    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature column names
        label_col: Label column name
        sequence_length: Sequence length for models
        train_split: Train/validation split ratio

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Extract features and labels
    features = df[feature_cols].values
    labels = df[label_col].values

    # Train/validation split
    split_idx = int(len(features) * train_split)

    train_features = features[:split_idx]
    train_labels = labels[:split_idx]

    val_features = features[split_idx:]
    val_labels = labels[split_idx:]

    # Create datasets
    train_dataset = OrderBookDataset(train_features, train_labels, sequence_length)
    val_dataset = OrderBookDataset(val_features, val_labels, sequence_length)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    return train_loader, val_loader


def main():
    """Main execution function"""
    logger.info("Starting deep learning model training")

    # Example date and instrument
    example_date = "2025-09-15"
    example_instrument = "AAPL.P.XNAS"

    # Load features
    feature_file = FEATURES_PATH / f"date={example_date}" / f"{example_instrument}.parquet"

    if not feature_file.exists():
        logger.error(f"Feature file not found: {feature_file}")
        logger.info("Please run 1_feature_engineering.py first")
        return

    df = read_parquet(feature_file)
    logger.info(f"Loaded features: {df.shape}")

    # Select feature columns (exclude metadata and label)
    exclude_cols = ["instrument_id", "venue", "ts_event", "book_seq", "label"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    logger.info(f"Using {len(feature_cols)} features")

    # Prepare data
    train_loader, val_loader = prepare_data(
        df, feature_cols,
        sequence_length=DEEPLOB_CONFIG["sequence_length"]
    )

    # Initialize DeepLOB model
    model_deeplob = DeepLOB(
        n_features=len(feature_cols),
        n_classes=DEEPLOB_CONFIG["n_classes"],
        hidden_dims=DEEPLOB_CONFIG["hidden_dims"],
        kernel_sizes=DEEPLOB_CONFIG["kernel_sizes"],
        dropout=DEEPLOB_CONFIG["dropout"]
    )

    # Train DeepLOB
    logger.info("Training DeepLOB model")
    trainer = ModelTrainer(model_deeplob)
    trainer.setup_optimizer(
        learning_rate=TRAINING_CONFIG["learning_rate"],
        optimizer_type=TRAINING_CONFIG["optimizer"]
    )

    history = trainer.train(
        train_loader, val_loader,
        epochs=TRAINING_CONFIG["epochs"],
        early_stopping_patience=TRAINING_CONFIG["early_stopping_patience"]
    )

    # Save model
    model_path = MODELS_PATH / "deeplob_model.pth"
    trainer.save_model(model_path)

    logger.info("Deep learning model training complete")


if __name__ == "__main__":
    main()
