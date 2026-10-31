"""PyTorch Baseline Convolutional Neural Network (CNN) Architecture.

Architecture:
    Input (batch_size, 3, 224, 224)
    Block 1: Conv2d(3 -> 32) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)
    Block 2: Conv2d(32 -> 64) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)
    Block 3: Conv2d(64 -> 128) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)
    Block 4: Conv2d(128 -> 256) -> BatchNorm2d -> ReLU
    Top:     AdaptiveAvgPool2d((1, 1)) -> Flatten -> Linear(256 -> 128) -> ReLU -> Dropout(0.5) -> Linear(128 -> 7)

Note: Raw logits are outputted (no Softmax layer) for compatibility with nn.CrossEntropyLoss.

Runnable via:
    python -m src.model
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


class SkinLesionCNN(nn.Module):
    """PyTorch Baseline CNN for Skin Lesion Classification."""

    def __init__(self, num_classes: int = 7):
        super(SkinLesionCNN, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Classification Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


def build_model(num_classes: int = 7) -> nn.Module:
    """Build uncompiled PyTorch baseline CNN model."""
    return SkinLesionCNN(num_classes=num_classes)


def validate_model() -> None:
    """Validate PyTorch CNN model architecture and test forward pass on 1 batch."""
    print("========================================")
    print("PYTORCH MODEL VALIDATION")
    print("========================================")
    model = build_model()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    val_csv = Path("data/processed/val.csv")
    if not val_csv.exists():
        print("[NOTICE] val.csv not found. Running dummy tensor forward pass.")
        dummy_input = torch.randn(32, 3, 224, 224).to(device)
        with torch.no_grad():
            outputs = model(dummy_input)
    else:
        from src.data_pipeline import create_dataloader

        val_loader = create_dataloader(val_csv, batch_size=32, shuffle=False, num_workers=0)
        val_images, val_labels = next(iter(val_loader))
        val_images = val_images.to(device)
        with torch.no_grad():
            outputs = model(val_images)

    print(f"\nDevice Selected:      {device.type.upper()}")
    print(f"Input Shape:          ({val_images.shape[0] if val_csv.exists() else 32}, 3, 224, 224)")
    print(f"Output Logits Shape:  {outputs.shape}")
    print("========================================")


if __name__ == "__main__":
    validate_model()
