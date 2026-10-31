# Dataset Directory Structure

This directory stores the HAM10000 skin lesion dataset for the classification pipeline.

## Structure

- `raw/`: Raw downloaded images and metadata CSV (`HAM10000_metadata.csv`).
- `processed/`: Processed dataset splits (train, validation, test CSVs and cached arrays).

## Source Dataset

- **Dataset**: HAM10000 ("Human Against Machine with 10000 training images")
- **Kaggle URL**: [https://www.kaggle.com/datasets/ahmedxc4/skin-ds](https://www.kaggle.com/datasets/ahmedxc4/skin-ds)
- **Categories (7)**:
  - `akiec`: Actinic Keratosis / Intraepithelial Carcinoma
  - `bcc`: Basal Cell Carcinoma
  - `bkl`: Benign Keratosis-like Lesions
  - `df`: Dermatofibroma
  - `mel`: Melanoma
  - `nv`: Melanocytic Nevus
  - `vasc`: Vascular Lesions

> **Note**: Raw image files in `data/raw/` and processed splits in `data/processed/` are ignored by git (`.gitignore`).
