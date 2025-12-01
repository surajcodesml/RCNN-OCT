"""Inference utilities for OCT object detection."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import _extract_arrays, _load_pickle
from model import build_model


def load_model(checkpoint_path: Path, num_classes: int | None = None, device: torch.device | None = None) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    label_mapping: Dict[int, int] = checkpoint.get("label_mapping", {})
    if num_classes is None:
        num_classes = len(label_mapping) + 1 if label_mapping else 2
    model = build_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device or torch.device("cpu"))
    model.eval()
    return model


def run_inference(model: torch.nn.Module, sample_path: Path, device: torch.device | None = None, score_threshold: float = 0.5):
    sample = _load_pickle(sample_path)
    image_np, _, _ = _extract_arrays(sample)
    image_tensor = torch.from_numpy(np.asarray(image_np, dtype=np.float32))
    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0)
    elif image_tensor.ndim == 3 and image_tensor.shape[0] != 1:
        image_tensor = image_tensor.permute(2, 0, 1)
    if image_tensor.shape[0] == 1:
        image_tensor = image_tensor.repeat(3, 1, 1)
    image_tensor = image_tensor / 255.0

    model_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(model_device)

    with torch.no_grad():
        outputs = model([image_tensor.to(model_device)])
    output = outputs[0]
    scores = output["scores"].cpu()
    keep = scores >= score_threshold
    boxes = output["boxes"].cpu()[keep]
    labels = output["labels"].cpu()[keep]
    scores = scores[keep]
    return image_tensor.cpu(), boxes, scores, labels


def visualize_predictions(image: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, save_path: Path | None = None) -> None:
    """Visualize predictions on a single image."""
    if image.dim() == 3 and image.shape[0] == 3:
        image_np = image[0].cpu().numpy()
    else:
        image_np = image.squeeze().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(image_np, cmap="gray")
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], label=f"{int(label.item())}:{score:.2f}")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single OCT pickle file")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--save-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device=device)
    image, boxes, scores, labels = run_inference(model, args.sample, device=device, score_threshold=args.score_threshold)
    visualize_predictions(image, boxes, scores, labels, save_path=args.save_path)


if __name__ == "__main__":
    main()
