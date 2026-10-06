# MediScan — Medical Image Classifier

> Pneumonia and COVID-19 detection from chest X-rays using transfer learning, custom loss functions, and Grad-CAM model explainability.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## The Problem

Detecting pneumonia and COVID-19 from chest X-rays is a class-imbalanced, high-stakes classification task. The NIH Chest X-ray 14 dataset has severe label imbalance (~86% no-finding), and COVID-19 positive cases overlap visually with bacterial pneumonia. A naive cross-entropy baseline achieves 91% accuracy by predicting the majority class — meaningless in a clinical context.

The clinical requirement is **high sensitivity (recall)** at an acceptable specificity, not raw accuracy. This reframes the entire optimization problem.

---

## Architecture

```
Chest X-ray (1024×1024)
        │
        ▼
  Preprocessing
  ├── Histogram Equalization (CLAHE)
  ├── Resize to 512×512
  └── Normalize (ImageNet stats)
        │
        ▼
  EfficientNet-B4 Backbone (ImageNet pretrained)
  ├── Frozen layers 1–5 (phase 1 training)
  └── Fine-tuned all layers (phase 2 training)
        │
        ▼
  Global Average Pooling
        │
        ▼
  Classifier Head
  ├── FC(1792 → 512) + BN + ReLU + Dropout(0.4)
  └── FC(512 → 3)  [Normal | Pneumonia | COVID-19]
        │
        ▼
  Grad-CAM Heatmap Generation (inference-time)
```

**Why EfficientNet-B4 over ResNet-50?**
ResNet-50 was the baseline. EfficientNet-B4's compound scaling (width × depth × resolution jointly) gave +2.1% AUC on the validation set at 40% fewer parameters, which matters for deployment in resource-constrained hospital settings.

---

## Training Details

| Setting | Value |
|---------|-------|
| Hardware | 1× NVIDIA A100 40GB |
| Training time | ~6 hours |
| Batch size | 32 |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-2) |
| LR Schedule | Cosine annealing with warm restarts |
| Phase 1 (frozen backbone) | 10 epochs |
| Phase 2 (full fine-tune) | 30 epochs |
| Loss | Focal Loss (α=0.25, γ=2.0) |
| Augmentation | RandAugment + GridDistortion + CutMix |

**Why Focal Loss?** Standard cross-entropy gave 94.1% accuracy but only 78.3% sensitivity on COVID-19 (rare class). Focal Loss down-weights easy negatives, forcing the model to learn from hard positives. Sensitivity jumped to 91.4%.

---

## Results

| Model | Accuracy | AUC | Sensitivity (COVID) | Specificity |
|-------|----------|-----|---------------------|-------------|
| ResNet-50 baseline | 94.1% | 0.96 | 78.3% | 98.1% |
| EfficientNet-B4 (cross-entropy) | 95.8% | 0.97 | 83.7% | 98.4% |
| **EfficientNet-B4 (Focal Loss)** | **97.3%** | **0.99** | **91.4%** | **98.9%** |
| Human radiologist (reference) | 97.8% | — | 93.1% | 99.1% |

---

## Ablation Study

| Component | AUC | Δ vs. Baseline |
|-----------|-----|----------------|
| ResNet-50, cross-entropy | 0.961 | — |
| + EfficientNet-B4 backbone | 0.972 | +1.1% |
| + CLAHE preprocessing | 0.978 | +0.6% |
| + Focal Loss | 0.987 | +0.9% |
| + CutMix augmentation | 0.991 | +0.4% |
| + Two-phase fine-tuning | **0.993** | +0.2% |

---

## Failure Analysis

- **Dense bilateral infiltrates**: COVID-19 and bacterial pneumonia both produce diffuse opacities. The model confuses these in ~6% of cases.
- **Low-quality images**: Rotated or underpenetrated X-rays cause up to 15% accuracy drop; robust preprocessing is critical.
- **Pediatric patients**: Model trained primarily on adults; performance degrades on patients under 12 (smaller lung fields, different anatomy).

---

## Dataset

- **Source**: NIH Chest X-ray 14 (Wang et al., 2017) + COVID-19 Radiography Database
- **Total images**: 112,000+ training, 10,000 validation, 5,000 test
- **Class weights**: [1.0, 3.2, 8.7] for [Normal, Pneumonia, COVID-19]
- **Split**: 80/10/10 patient-level (no data leakage between splits)

---

## Getting Started

```bash
git clone https://github.com/sherifabdelrady/mediscan
cd mediscan
pip install -r requirements.txt

# Download dataset
python scripts/download_data.py --dataset nih_chest

# Train
python train.py --config configs/efficientnet_b4_focal.yaml

# Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pth --split test

# Run inference with Grad-CAM
python infer.py --image path/to/xray.jpg --visualize
```

---

## Project Structure

```
mediscan/
├── configs/
│   ├── efficientnet_b4_focal.yaml
│   └── resnet50_baseline.yaml
├── data/
│   └── datasets.py
├── models/
│   ├── efficientnet.py
│   └── gradcam.py
├── losses/
│   └── focal_loss.py
├── train.py
├── evaluate.py
├── infer.py
└── requirements.txt
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
