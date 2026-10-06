"""
MediScan — Quick inference demo
Runs without training. Downloads a sample X-ray from NIH OpenI and classifies it.
Usage: python demo.py [--image path/to/xray.png]
"""

import torch
import torchvision.transforms as T
import torchvision.models as models
import argparse
import urllib.request
import os
from pathlib import Path
from PIL import Image

LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "No Finding", "Nodule", "Pleural Thickening", "Pneumonia",
    "Pneumothorax",
]

SAMPLE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/"
    "Chest_radiograph_in_influenza_and_Haemophilus_influenzae%2C_PA%2C_inverted.jpg"
    "/400px-Chest_radiograph_in_influenza_and_Haemophilus_influenzae%2C_PA%2C_inverted.jpg"
)

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(num_classes: int = 15) -> torch.nn.Module:
    """DenseNet-121 with custom multi-label head (mirrors train.py)."""
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(in_features, num_classes),
        torch.nn.Sigmoid(),
    )
    return model


@torch.inference_mode()
def predict(model: torch.nn.Module, image_path: str, threshold: float = 0.4):
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)
    probs = model(tensor).squeeze().numpy()
    results = [(LABELS[i], float(probs[i])) for i in range(len(LABELS))]
    results.sort(key=lambda x: -x[1])
    print(f"\nPredictions for: {image_path}")
    print("-" * 40)
    for label, prob in results[:5]:
        bar = "█" * int(prob * 20)
        flag = " ← POSITIVE" if prob >= threshold else ""
        print(f"  {label:<22} {prob:.3f}  {bar}{flag}")
    print(f"\n(Threshold={threshold} | ImageNet-pretrained weights — not fine-tuned)")
    return results


def main():
    parser = argparse.ArgumentParser(description="MediScan Quick Demo")
    parser.add_argument("--image", default=None, help="Path to chest X-ray image")
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()

    # Download sample image if none provided
    img_path = args.image
    if img_path is None:
        img_path = "sample_xray.jpg"
        if not os.path.exists(img_path):
            print("Downloading sample X-ray...")
            urllib.request.urlretrieve(SAMPLE_URL, img_path)
            print(f"Saved to {img_path}")

    print("Loading DenseNet-121 (ImageNet pretrained)...")
    model = build_model()
    model.eval()

    predict(model, img_path, args.threshold)
    print("\nNote: For real predictions, train with train.py on NIH ChestX-ray14 dataset.")


if __name__ == "__main__":
    main()
