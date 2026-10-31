"""Minimal PyTorch Image Data Pipeline for Skin Lesion Classification.

Provides PyTorch Dataset and DataLoader creation with PIL image loading,
224x224 resizing, [0.0, 1.0] float32 tensor conversion (3, 224, 224),
and integer label encoding (0..6).

Runnable via:
    python -m src.data_pipeline
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_SIZE: Tuple[int, int] = (224, 224)
NUM_CLASSES: int = 7

# Fixed integer label mapping for 7 HAM10000 categories
LABEL_MAP: Dict[str, int] = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}

REV_LABEL_MAP: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}


class SkinLesionDataset(Dataset):
    """PyTorch Dataset for loading and preprocessing skin lesion images."""

    def __init__(self, csv_path: Path):
        """Initialize dataset from CSV manifest.

        Args:
            csv_path: Path to train.csv, val.csv, or test.csv.
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self.image_paths = self.df["image_path"].values
        self.labels = [LABEL_MAP[l] for l in self.df["label"].values]

        # PyTorch image transformation pipeline: Resize to 224x224 and convert to float tensor [0.0, 1.0] (3, 224, 224)
        self.transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def create_dataloader(
    csv_path: Path,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    """Create PyTorch DataLoader for train, val, or test splits.

    Args:
        csv_path: Path to split CSV file.
        batch_size: Batch size (default 32).
        shuffle: Whether to shuffle data (True for training, False otherwise).
        num_workers: Subprocess worker count for data loading.

    Returns:
        PyTorch DataLoader instance.
    """
    dataset = SkinLesionDataset(csv_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def validate_pipeline() -> None:
    """Validate PyTorch data pipeline by loading 1 batch from train.csv."""
    train_csv = Path("data/processed/train.csv")
    if not train_csv.exists():
        print("[ERROR] train.csv not found. Please complete Phase 3 first.")
        return

    print("========================================")
    print("PYTORCH DATA PIPELINE VALIDATION")
    print("========================================")
    dataloader = create_dataloader(train_csv, batch_size=32, shuffle=True, num_workers=0)
    images, labels = next(iter(dataloader))

    print(f"Image Batch Shape: {images.shape}")
    print(f"Label Batch Shape: {labels.shape}")
    print(f"Image Dtype:       {images.dtype}")
    print(f"Label Dtype:       {labels.dtype}")
    print(f"Min Pixel Value:   {images.min().item():.4f}")
    print(f"Max Pixel Value:   {images.max().item():.4f}")
    print(f"Min Label Value:   {labels.min().item()}")
    print(f"Max Label Value:   {labels.max().item()}")
    print("========================================")


if __name__ == "__main__":
    validate_pipeline()
