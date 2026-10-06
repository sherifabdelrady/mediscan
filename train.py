"""
MediScan — Medical Image Classification
Multi-label chest X-ray classifier trained on NIH Chest X-Ray 14.
Architecture: DenseNet-121 backbone with focal loss for class imbalance.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import argparse

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NUM_CLASSES = len(LABELS)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class FocalLoss(nn.Module):
    """Binary focal loss for multi-label classification."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        p_t = torch.exp(-bce)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class ChestXRayDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["Image Index"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label_str = row["Finding Labels"]
        target = torch.zeros(NUM_CLASSES)
        for finding in label_str.split("|"):
            if finding in LABELS:
                target[LABELS.index(finding)] = 1.0
        return img, target


def build_model(pretrained=True, freeze_backbone=False):
    model = models.densenet121(pretrained=pretrained)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )
    return model


def get_transforms(split="train"):
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


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    from sklearn.metrics import roc_auc_score
    aucs = []
    for i in range(NUM_CLASSES):
        if targets[:, i].sum() > 0:
            aucs.append(roc_auc_score(targets[:, i], preds[:, i]))
    return np.mean(aucs)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_ds = ChestXRayDataset(args.train_csv, args.img_dir, get_transforms("train"))
    val_ds   = ChestXRayDataset(args.val_csv,   args.img_dir, get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = build_model(pretrained=True, freeze_backbone=False).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.cuda.amp.GradScaler()

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        auc  = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Mean AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), args.output / "mediscan_best.pt")
            print(f"  → Saved best model (AUC {best_auc:.4f})")

    print(f"\nTraining complete. Best Mean AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv",  type=Path, required=True)
    parser.add_argument("--val-csv",    type=Path, required=True)
    parser.add_argument("--img-dir",    type=Path, required=True)
    parser.add_argument("--output",     type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs",     type=int,  default=30)
    parser.add_argument("--batch-size", type=int,  default=64)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    main(args)
