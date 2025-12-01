"""Training and evaluation script for Faster R-CNN on OCT B-scans."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from dataset import OCTDetectionDataset, create_datasets, detection_collate_fn
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on OCT data")
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory containing .pkl files")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--score-threshold", type=float, default=0.5)
    return parser.parse_args()


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                scores = output["scores"].cpu().numpy()
                keep = scores >= score_threshold
                pred_boxes = output["boxes"].cpu().numpy()[keep]
                gt_boxes = target["boxes"].cpu().numpy()

                matched_gt = set()
                for pred_box in pred_boxes:
                    ious = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
                    if not ious:
                        fp += 1
                        continue
                    best_idx = int(np.argmax(ious))
                    if ious[best_idx] >= 0.5 and best_idx not in matched_gt:
                        tp += 1
                        matched_gt.add(best_idx)
                    else:
                        fp += 1
                fn += max(len(gt_boxes) - len(matched_gt), 0)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    loss_accum = {"loss": 0.0, "loss_classifier": 0.0, "loss_box_reg": 0.0, "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0}
    num_batches = 0
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        for key in loss_accum:
            loss_accum[key] += loss_dict.get(key, torch.tensor(0.0, device=device)).item()
        num_batches += 1

    for key in loss_accum:
        loss_accum[key] /= max(num_batches, 1)
    return loss_accum


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, label_mapping = create_datasets(args.data_root, val_ratio=args.val_ratio, seed=args.seed)
    num_classes = len(label_mapping) + 1 if label_mapping else 2

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=detection_collate_fn)

    model = build_model(num_classes=num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_path = args.output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        loss_dict = train_one_epoch(model, optimizer, train_loader, device)
        lr_scheduler.step()

        val_metrics = evaluate(model, val_loader, device, score_threshold=args.score_threshold)

        print(
            f"Epoch {epoch}: loss={loss_dict['loss']:.4f}, "
            f"cls={loss_dict['loss_classifier']:.4f}, box={loss_dict['loss_box_reg']:.4f}, "
            f"obj={loss_dict['loss_objectness']:.4f}, rpn_box={loss_dict['loss_rpn_box_reg']:.4f}, "
            f"precision={val_metrics['precision']:.4f}, recall={val_metrics['recall']:.4f}, f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({"model_state": model.state_dict(), "label_mapping": label_mapping}, best_path)
            print(f"Saved new best model to {best_path} with F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
