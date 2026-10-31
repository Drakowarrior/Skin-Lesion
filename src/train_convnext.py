"""
Phase 10A: ConvNeXt-Small Multiclass Challenger Training & Validation Script.
Strictly isolated from production ResNet50 model and test set.
"""

import os
import sys
import time
import json
import math
import hashlib
import platform
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support

# ---------------------------------------------------------
# 1. Reproducibility & Device Configuration
# ---------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

# ---------------------------------------------------------
# 2. Canonical Class Mapping
# ---------------------------------------------------------
CANONICAL_CLASSES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Chickenpox",
    "Cowpox",
    "Dermatofibroma",
    "HFMD",
    "Healthy",
    "Measles",
    "Melanocytic nevi",
    "Melanoma",
    "Monkeypox",
    "Squamous cell carcinoma",
    "Vascular lesions"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
NUM_CLASSES = len(CANONICAL_CLASSES)

# ---------------------------------------------------------
# 3. Dataset Class
# ---------------------------------------------------------
class SkinLesionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label_str = row['label']
        label_idx = CLASS_TO_IDX[label_str]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image at {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, label_idx

# ---------------------------------------------------------
# 4. Class-Balanced Focal Loss
# ---------------------------------------------------------
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_weights, gamma=2.0):
        super().__init__()
        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: (N, C), targets: (N,)
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
        # p_t: probability assigned to the ground truth class
        p_t = (probs * targets_one_hot).sum(dim=-1)
        p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)
        
        # focal weight
        focal_weight = torch.pow(1.0 - p_t, self.gamma)
        
        # class weight for the target class
        w_t = (self.class_weights * targets_one_hot).sum(dim=-1)
        
        loss = - w_t * focal_weight * torch.log(p_t)
        return loss.mean()

# ---------------------------------------------------------
# 5. Model Architecture Assembly
# ---------------------------------------------------------
def create_convnext_challenger(num_classes=14, dropout_rate=0.3):
    weights = models.ConvNeXt_Small_Weights.DEFAULT
    model = models.convnext_small(weights=weights)
    in_features = model.classifier[2].in_features
    
    # Replace the classification linear projection while preserving LayerNorm2d and Flatten
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    return model

# ---------------------------------------------------------
# 6. Evaluation Function
# ---------------------------------------------------------
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * images.size(0)

            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    avg_loss = total_loss / len(dataloader.dataset)
    top1_acc = accuracy_score(all_targets, all_preds)
    
    # Top-3 Accuracy
    top3_correct = 0
    for i in range(len(all_targets)):
        top3_indices = np.argsort(all_probs[i])[-3:]
        if all_targets[i] in top3_indices:
            top3_correct += 1
    top3_acc = top3_correct / len(all_targets)

    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    per_class_results = {}
    for i, c in enumerate(CANONICAL_CLASSES):
        per_class_results[c] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i])
        }

    melanoma_recall = per_class_results["Melanoma"]["recall"]

    return {
        "loss": float(avg_loss),
        "top1_accuracy": float(top1_acc),
        "top3_accuracy": float(top3_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "melanoma_recall": float(melanoma_recall),
        "per_class": per_class_results,
        "preds": all_preds,
        "targets": all_targets,
        "probs": all_probs
    }

# ---------------------------------------------------------
# 7. Main Execution Pipeline
# ---------------------------------------------------------
def main():
    print("=" * 70)
    print("PHASE 10A: CONVNEXT-SMALL MULTICLASS CHALLENGER TRAINING")
    print("=" * 70)

    # Pre-training Safety Verification: Production Model Checksum
    prod_model_path = "models/production/resnet50_skin_lesion_v1.keras"
    expected_prod_sha256 = "93a121c51900dc70d9e95b5a1d75e54c442fcdb4f20a1397625bcffb6dedd58e"
    
    with open(prod_model_path, "rb") as f:
        actual_prod_sha256 = hashlib.sha256(f.read()).hexdigest()
    
    print(f"Pre-check: Production Model SHA-256: {actual_prod_sha256}")
    if actual_prod_sha256 != expected_prod_sha256:
        print(f"CRITICAL ERROR: Production model checksum mismatch! Expected: {expected_prod_sha256}, Got: {actual_prod_sha256}")
        sys.exit(1)

    # Directories
    exp_dir = "models/experiments/phase10a_convnext"
    audit_dir = "audits/phase10a"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(audit_dir, exist_ok=True)

    # Load Manifests (Train & Val ONLY)
    train_csv = "data/processed/train.csv"
    val_csv = "data/processed/val.csv"
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"Loaded Train Samples: {len(train_df)} | Val Samples: {len(val_df)}")
    print("CONFIRMATION: Historical test set (test.csv) is strictly excluded and untouched.")

    # Calculate Class Weights from Train Split ONLY
    train_counts = {c: int(sum(train_df['label'] == c)) for c in CANONICAL_CLASSES}
    beta = 0.9999
    ens_weights = {}
    for c in CANONICAL_CLASSES:
        eff_num = (1.0 - (beta ** train_counts[c])) / (1.0 - beta)
        ens_weights[c] = 1.0 / eff_num

    mean_ens = np.mean(list(ens_weights.values()))
    for c in CANONICAL_CLASSES:
        ens_weights[c] = ens_weights[c] / mean_ens

    # Apply 2.5x Melanoma Clinical Sensitivity Multiplier
    ens_weights_with_mel = ens_weights.copy()
    ens_weights_with_mel['Melanoma'] = ens_weights_with_mel['Melanoma'] * 2.5
    mean_w = np.mean(list(ens_weights_with_mel.values()))
    for c in CANONICAL_CLASSES:
        ens_weights_with_mel[c] = float(ens_weights_with_mel[c] / mean_w)

    class_weights_list = [ens_weights_with_mel[c] for c in CANONICAL_CLASSES]
    
    # Save Class Weights Artifact
    with open(os.path.join(audit_dir, "class_weights.json"), "w") as f:
        json.dump({
            "strategy": "Effective Number of Samples (ENS) with beta=0.9999 and 2.5x Melanoma Multiplier",
            "train_counts": train_counts,
            "normalized_weights": {c: ens_weights_with_mel[c] for c in CANONICAL_CLASSES},
            "formula": "L = - w_c * (1 - p_t)^gamma * log(p_t), gamma=2.0"
        }, f, indent=2)

    # Transforms (384x384 resolution)
    train_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Save Preprocessing Config Artifact
    with open(os.path.join(audit_dir, "preprocessing_config.json"), "w") as f:
        json.dump({
            "input_resolution": [384, 384, 3],
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "color_space": "RGB"
            },
            "train_augmentations": [
                "Resize(384, 384)",
                "RandomHorizontalFlip(p=0.5)",
                "RandomVerticalFlip(p=0.5)",
                "RandomRotation(15 deg)",
                "ColorJitter(b=0.1, c=0.1, s=0.1)"
            ],
            "val_augmentations": [
                "Resize(384, 384)"
            ]
        }, f, indent=2)

    train_dataset = SkinLesionDataset(train_df, transform=train_transform)
    val_dataset = SkinLesionDataset(val_df, transform=val_transform)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    # Save Environment Artifact
    with open(os.path.join(audit_dir, "environment.json"), "w") as f:
        json.dump({
            "os": platform.platform(),
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "device": str(device),
            "batch_size": batch_size,
            "random_seed": SEED,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }, f, indent=2)

    # Model & Criterion
    model = create_convnext_challenger(num_classes=NUM_CLASSES, dropout_rate=0.3).to(device)
    criterion = ClassBalancedFocalLoss(class_weights=class_weights_list, gamma=2.0).to(device)

    # Training Hyperparameters
    stage_a_epochs = 3
    stage_b_epochs = 5
    total_epochs = stage_a_epochs + stage_b_epochs

    # Save Training Config Artifact
    with open(os.path.join(audit_dir, "training_config.json"), "w") as f:
        json.dump({
            "model_architecture": "ConvNeXt-Small (torchvision ImageNet-1k pretrained)",
            "total_params": sum(p.numel() for p in model.parameters()),
            "input_resolution": "384x384x3",
            "batch_size": batch_size,
            "stage_a_epochs": stage_a_epochs,
            "stage_b_epochs": stage_b_epochs,
            "stage_a_lr": 1e-3,
            "stage_b_lr": 1e-4,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "loss_function": "ClassBalancedFocalLoss (gamma=2.0, ENS beta=0.9999, Mel 2.5x)",
            "checkpoint_selection_metric": "Validation Macro F1"
        }, f, indent=2)

    print("\nStarting Stage A: Training Classification Head (Backbone Frozen)...")
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer_a = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    epoch_logs = []
    best_macro_f1 = -1.0
    best_epoch = -1
    best_checkpoint_path = os.path.join(exp_dir, "convnext_small_best.pt")

    start_time_total = time.time()

    # -----------------------------------------------------
    # STAGE A: Head Warmup
    # -----------------------------------------------------
    for epoch in range(1, stage_a_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer_a.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            
            if not math.isfinite(loss.item()):
                print(f"[ERROR] Loss is non-finite at Epoch {epoch}, Step {step}: {loss.item()}")
                sys.exit(1)

            loss.backward()
            optimizer_a.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        epoch_duration = time.time() - epoch_start

        # Validation
        val_results = evaluate_model(model, val_loader, criterion, device)
        
        print(f"[Stage A | Epoch {epoch}/{total_epochs}] Duration: {epoch_duration:.1f}s | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_results['loss']:.4f} | Val Top-1: {val_results['top1_accuracy']*100:.2f}% | "
              f"Val Macro F1: {val_results['macro_f1']*100:.2f}% | Mel Recall: {val_results['melanoma_recall']*100:.2f}%")

        log_row = {
            "epoch": epoch,
            "stage": "A",
            "lr": 1e-3,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_results["loss"],
            "val_top1_acc": val_results["top1_accuracy"],
            "val_top3_acc": val_results["top3_accuracy"],
            "val_macro_f1": val_results["macro_f1"],
            "val_weighted_f1": val_results["weighted_f1"],
            "val_melanoma_recall": val_results["melanoma_recall"],
            "duration_sec": epoch_duration
        }
        epoch_logs.append(log_row)

        if val_results["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_results["macro_f1"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_results,
                "class_mapping": CANONICAL_CLASSES
            }, best_checkpoint_path)

    # -----------------------------------------------------
    # STAGE B: Backbone Fine-Tuning
    # -----------------------------------------------------
    print("\nStarting Stage B: Fine-Tuning Backbone + Classification Head...")
    # Unfreeze top feature stages (stage 6 & 7 of features) and head
    for param in model.features.parameters():
        param.requires_grad = True

    optimizer_b = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 5e-5},
        {'params': model.classifier.parameters(), 'lr': 2e-4}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=stage_b_epochs, eta_min=1e-6)

    for epoch_b in range(1, stage_b_epochs + 1):
        epoch = stage_a_epochs + epoch_b
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer_b.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)

            if not math.isfinite(loss.item()):
                print(f"[ERROR] Loss is non-finite at Epoch {epoch}, Step {step}: {loss.item()}")
                sys.exit(1)

            loss.backward()
            optimizer_b.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total
        epoch_duration = time.time() - epoch_start

        # Validation
        val_results = evaluate_model(model, val_loader, criterion, device)
        current_lr = scheduler.get_last_lr()[0]

        print(f"[Stage B | Epoch {epoch}/{total_epochs}] Duration: {epoch_duration:.1f}s | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_results['loss']:.4f} | Val Top-1: {val_results['top1_accuracy']*100:.2f}% | "
              f"Val Macro F1: {val_results['macro_f1']*100:.2f}% | Mel Recall: {val_results['melanoma_recall']*100:.2f}%")

        log_row = {
            "epoch": epoch,
            "stage": "B",
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_results["loss"],
            "val_top1_acc": val_results["top1_accuracy"],
            "val_top3_acc": val_results["top3_accuracy"],
            "val_macro_f1": val_results["macro_f1"],
            "val_weighted_f1": val_results["weighted_f1"],
            "val_melanoma_recall": val_results["melanoma_recall"],
            "duration_sec": epoch_duration
        }
        epoch_logs.append(log_row)

        if val_results["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_results["macro_f1"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_results,
                "class_mapping": CANONICAL_CLASSES
            }, best_checkpoint_path)

    total_duration_sec = time.time() - start_time_total
    print(f"\nTraining Complete in {total_duration_sec/60:.2f} minutes. Best Epoch: {best_epoch} with Val Macro F1: {best_macro_f1*100:.2f}%")

    # -----------------------------------------------------
    # 8. Save Metrics CSV & Best Checkpoint Evaluation
    # -----------------------------------------------------
    epoch_df = pd.DataFrame(epoch_logs)
    epoch_df.to_csv(os.path.join(audit_dir, "epoch_metrics.csv"), index=False)

    # Load Best Checkpoint for Final Validation Report
    print(f"\nEvaluating Best Checkpoint from {best_checkpoint_path} (Epoch {best_epoch})...")
    ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Latency Benchmark (Single Image on Device)
    model.eval()
    dummy_input = torch.randn(1, 3, 384, 384, device=device)
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    if device.type == 'mps':
        torch.mps.synchronize()

    latency_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == 'mps':
            torch.mps.synchronize()
        latency_times.append((time.perf_counter() - t0) * 1000.0) # ms
    
    mean_latency_ms = float(np.mean(latency_times))
    p95_latency_ms = float(np.percentile(latency_times, 95))
    print(f"Inference Latency: Mean = {mean_latency_ms:.2f} ms | P95 = {p95_latency_ms:.2f} ms")

    # Full Validation Run
    final_val = evaluate_model(model, val_loader, criterion, device)

    # Save validation_metrics.json
    val_metrics_payload = {
        "best_epoch": int(best_epoch),
        "val_loss": float(final_val["loss"]),
        "top1_accuracy": float(final_val["top1_accuracy"]),
        "top3_accuracy": float(final_val["top3_accuracy"]),
        "macro_f1": float(final_val["macro_f1"]),
        "weighted_f1": float(final_val["weighted_f1"]),
        "melanoma_recall": float(final_val["melanoma_recall"]),
        "mean_latency_ms": float(mean_latency_ms),
        "p95_latency_ms": float(p95_latency_ms),
        "total_training_duration_minutes": float(total_duration_sec / 60.0)
    }
    with open(os.path.join(audit_dir, "validation_metrics.json"), "w") as f:
        json.dump(val_metrics_payload, f, indent=2)

    # Save classification_report.json
    with open(os.path.join(audit_dir, "classification_report.json"), "w") as f:
        json.dump(final_val["per_class"], f, indent=2)

    # Compute Checkpoint SHA-256
    with open(best_checkpoint_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    with open(os.path.join(audit_dir, "checkpoint_integrity.json"), "w") as f:
        json.dump({
            "checkpoint_path": best_checkpoint_path,
            "checkpoint_sha256": ckpt_sha256,
            "best_epoch": best_epoch,
            "filesize_bytes": os.path.getsize(best_checkpoint_path),
            "production_model_sha256": actual_prod_sha256,
            "production_model_verified_intact": True
        }, f, indent=2)

    # -----------------------------------------------------
    # 9. Plotting: Confusion Matrix & Training Curves
    # -----------------------------------------------------
    # Confusion Matrix Plot
    cm = confusion_matrix(final_val["targets"], final_val["preds"], labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig, ax = plt.subplots(figsize=(14, 12))
    cax = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_title(f'Phase 10A ConvNeXt-Small Validation Confusion Matrix (Epoch {best_epoch})', fontsize=14, pad=15)
    tick_marks = np.arange(NUM_CLASSES)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(CANONICAL_CLASSES, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(CANONICAL_CLASSES, fontsize=10)

    # Text annotations
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_norm[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(audit_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_df['epoch'], epoch_df['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epoch_df['epoch'], epoch_df['val_loss'], 'r-o', label='Val Loss')
    plt.title('Loss Curves (CB-Focal Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_df['epoch'], epoch_df['val_top1_acc'] * 100, 'g-o', label='Val Top-1 Acc (%)')
    plt.plot(epoch_df['epoch'], epoch_df['val_macro_f1'] * 100, 'm-o', label='Val Macro F1 (%)')
    plt.plot(epoch_df['epoch'], epoch_df['val_melanoma_recall'] * 100, 'c-o', label='Val Melanoma Recall (%)')
    plt.title('Validation Metrics Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(audit_dir, "training_curves.png"), dpi=200)
    plt.close()

    print(f"\nPhase 10A Challenger training & validation successfully finished!")
    print(f"Artifacts saved in {audit_dir}/ and checkpoint saved at {best_checkpoint_path}")

if __name__ == "__main__":
    main()
