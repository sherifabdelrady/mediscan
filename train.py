"""
MediScan - Medical Image Classification
Multi-label chest X-ray classifier trained on NIH Chest X-Ray 14.
Architecture: DenseNet-121 backbone with focal loss.

Audit fixes (audit/production-remediation):
- P2 FIX: models.densenet121(pretrained=True) is deprecated since PyTorch 0.13.
  Updated to use weights=DenseNet121_Weights.DEFAULT.
- P2 FIX: torch.cuda.amp.autocast() is deprecated since PyTorch 2.4.
  Updated to torch.amp.autocast('cuda').
- P2 FIX: torch.cuda.amp.GradScaler() deprecated. Updated to torch.amp.GradScaler('cuda').
- P2 FIX: README claimed EfficientNet-B4 — the implementation uses DenseNet-121.
  Code and docstring now explicitly document DenseNet-121.
- P2 FIX: args.output used as a Path for torch.save() target file, not directory.
  Added mkdir() with correct parent + explicit .pt filename.
- P3 FIX: evaluate() imported sklearn inside the function on every call.
  Moved to module-level import with try/except.
- CLARIFICATION: The reported metric is Mean ROC-AUC across 14 pathologies,
  NOT "97% accuracy". The README has been updated to reflect this correctly.

NOTE: No patient-level split enforcement in this script. The NIH Chest X-Ray 14
dataset's official train/val/test splits (train_val_list.txt / test_list.txt)
should be used to prevent patient-level data leakage.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torchvision.models import DenseNet121_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(LABELS)   # 14

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary focal loss for multi-label classification (Lin et al., 2017)."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce   = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        p_t   = torch.exp(-bce)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChestXRayDataset(Dataset):
    """
    NIH Chest X-Ray 14 dataset loader.

    CSV format expected:
      Image Index | Finding Labels | Patient ID | ... (extra columns ignored)

    Use the official train_val_list.txt / test_list.txt splits to prevent
    patient-level leakage. Pre-filter your CSV to contain only the intended
    patient subset before passing to this dataset.
    """

    def __init__(self, csv_path: str, img_dir: str, transform=None):
        import pandas as pd
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = Path(img_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["Image Index"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label_str = row["Finding Labels"]
        target    = torch.zeros(NUM_CLASSES)
        for finding in label_str.split("|"):
            finding = finding.strip()
            if finding in LABELS:
                target[LABELS.index(finding)] = 1.0
        return img, target


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(freeze_backbone: bool = False) -> nn.Module:
    """
    DenseNet-121 (ImageNet pretrained) with a custom multi-label head.

    Output: logits (B, 14) — apply sigmoid for probabilities.
    """
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features       = model.classifier.in_features
    model.classifier  = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(split: str = "train") -> T.Compose:
    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, criterion, device, scaler
) -> float:
    model.train()
    total_loss = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Evaluation — Mean ROC-AUC across 14 pathologies
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device) -> float:
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn is required for evaluation: pip install scikit-learn")
    model.eval()
    all_preds:   list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for imgs, targets in loader:
        imgs    = imgs.to(device)
        logits  = model(imgs)
        preds   = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())
    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    aucs = []
    for i in range(NUM_CLASSES):
        if targets[:, i].sum() > 0:
            aucs.append(roc_auc_score(targets[:, i], preds[:, i]))
    return float(np.mean(aucs)) if aucs else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Architecture: DenseNet-121  |  Task: {NUM_CLASSES}-class multi-label")
    print(f"Metric: Mean ROC-AUC across {NUM_CLASSES} pathologies")

    train_ds = ChestXRayDataset(args.train_csv, args.img_dir, get_transforms("train"))
    val_ds   = ChestXRayDataset(args.val_csv,   args.img_dir, get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = build_model(freeze_backbone=False).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt  = output_dir / "mediscan_best.pt"

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        auc  = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Mean ROC-AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ Saved best model (AUC {best_auc:.4f}) → {best_ckpt}")

    print(f"\nTraining complete. Best Mean ROC-AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediScan DenseNet-121 training")
    parser.add_argument("--train-csv",  type=Path, required=True,
                        help="CSV with 'Image Index' and 'Finding Labels' columns (train split)")
    parser.add_argument("--val-csv",    type=Path, required=True,
                        help="CSV for validation split")
    parser.add_argument("--img-dir",    type=Path, required=True,
                        help="Directory containing NIH CXR images")
    parser.add_argument("--output",     type=Path, default=Path("checkpoints"),
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
