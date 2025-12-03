"""Batch inference with comprehensive metrics for OCT object detection."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from tqdm import tqdm

from dataset import create_datasets_from_splits
from inference import load_model
from model import build_model


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
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


def compute_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """Compute Average Precision using 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def compute_metrics_at_iou(
    all_pred_boxes: List[np.ndarray],
    all_pred_scores: List[np.ndarray],
    all_gt_boxes: List[np.ndarray],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Compute detection metrics at a specific IoU threshold.
    
    Returns:
        Dictionary with precision, recall, f1, mAP, TP, FP, FN, TN
    """
    # Compute TP, FP, FN at IoU threshold
    tp = fp = fn = tn = 0
    total_preds = 0
    
    for pred_boxes, pred_scores, gt_boxes in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
        # Filter by score threshold
        keep = pred_scores >= score_threshold
        pred_boxes_filtered = pred_boxes[keep]
        
        total_preds += len(pred_boxes_filtered)
        
        matched_gt = set()
        for pred_box in pred_boxes_filtered:
            if len(gt_boxes) == 0:
                fp += 1
                continue
            
            ious = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
            best_idx = int(np.argmax(ious))
            
            if ious[best_idx] >= iou_threshold and best_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        
        fn += len(gt_boxes) - len(matched_gt)
        
        # TN: Images without GT boxes where we correctly predict nothing
        if len(gt_boxes) == 0 and len(pred_boxes_filtered) == 0:
            tn += 1
    
    # Compute mAP at this IoU threshold
    all_pred_data = []
    for pred_boxes, pred_scores, gt_boxes in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
        for box, score in zip(pred_boxes, pred_scores):
            all_pred_data.append((score, box, gt_boxes))
    
    all_pred_data.sort(key=lambda x: x[0], reverse=True)
    
    # Compute precision-recall curve
    tp_curve = 0
    fp_curve = 0
    precisions = []
    recalls = []
    total_gt = sum(len(gt) for gt in all_gt_boxes)
    
    matched_gts = [set() for _ in all_gt_boxes]
    
    for score, pred_box, gt_boxes in all_pred_data:
        # Find which image this prediction belongs to
        gt_idx = -1
        for i, gts in enumerate(all_gt_boxes):
            if np.array_equal(gt_boxes, gts):
                gt_idx = i
                break
        
        if len(gt_boxes) == 0:
            fp_curve += 1
        else:
            ious = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
            best_idx = int(np.argmax(ious))
            
            if ious[best_idx] >= iou_threshold and best_idx not in matched_gts[gt_idx]:
                tp_curve += 1
                matched_gts[gt_idx].add(best_idx)
            else:
                fp_curve += 1
        
        precision = tp_curve / (tp_curve + fp_curve) if (tp_curve + fp_curve) > 0 else 0.0
        recall = tp_curve / total_gt if total_gt > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
    
    # Compute mAP
    if len(precisions) > 0:
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        mAP = compute_ap(precisions, recalls)
    else:
        mAP = 0.0
    
    # Final metrics
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP": mAP,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_predictions": total_preds,
    }


def visualize_detection(
    image: torch.Tensor,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    save_path: Path,
    sample_name: str,
) -> None:
    """Visualize predictions vs ground truth."""
    # Convert image to numpy
    if image.dim() == 3 and image.shape[0] == 3:
        image_np = image[0].cpu().numpy()
    else:
        image_np = image.squeeze().cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot ground truth
    ax1.imshow(image_np, cmap="gray")
    ax1.set_title(f"Ground Truth - {sample_name}", fontsize=12)
    ax1.axis("off")
    
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="green", facecolor="none"
        )
        ax1.add_patch(rect)
        ax1.text(x1, y1 - 5, f"GT:{int(label)}", color="green", fontsize=10, weight="bold")
    
    # Plot predictions
    ax2.imshow(image_np, cmap="gray")
    ax2.set_title(f"Predictions - {sample_name}", fontsize=12)
    ax2.axis("off")
    
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax2.add_patch(rect)
        ax2.text(x1, y1 - 5, f"{int(label)}:{score:.2f}", color="red", fontsize=10, weight="bold")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def batch_inference(
    model: torch.nn.Module,
    test_dataset,
    device: torch.device,
    score_threshold: float,
    output_dir: Path,
    visualize_samples: int = 10,
) -> Dict:
    """
    Run inference on test dataset and compute comprehensive metrics.
    
    Args:
        model: Trained detection model
        test_dataset: Test dataset
        device: Computing device
        score_threshold: Score threshold for predictions
        output_dir: Directory to save results
        visualize_samples: Number of samples to visualize
    
    Returns:
        Dictionary with all metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    model.eval()
    
    # Collect all predictions and ground truths
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []
    all_images = []
    all_file_paths = []
    
    print(f"Running inference on {len(test_dataset)} test samples...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Processing"):
            image, target = test_dataset[idx]
            
            # Run inference
            image_batch = image.unsqueeze(0).to(device)
            outputs = model(image_batch)
            output = outputs[0]
            
            # Extract predictions
            pred_boxes = output["boxes"].cpu().numpy()
            pred_scores = output["scores"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
            
            # Extract ground truth
            gt_boxes = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()
            
            all_pred_boxes.append(pred_boxes)
            all_pred_scores.append(pred_scores)
            all_pred_labels.append(pred_labels)
            all_gt_boxes.append(gt_boxes)
            all_gt_labels.append(gt_labels)
            all_images.append(image)
            all_file_paths.append(test_dataset.files[idx])
    
    # Compute metrics at IoU=0.5 and IoU=0.5:0.95
    print("\nComputing metrics...")
    metrics_05 = compute_metrics_at_iou(
        all_pred_boxes, all_pred_scores, all_gt_boxes,
        iou_threshold=0.5, score_threshold=score_threshold
    )
    
    # Compute mAP@0.5:0.95 (average over IoU thresholds 0.5, 0.55, ..., 0.95)
    map_scores = []
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        metrics_temp = compute_metrics_at_iou(
            all_pred_boxes, all_pred_scores, all_gt_boxes,
            iou_threshold=iou_thresh, score_threshold=score_threshold
        )
        map_scores.append(metrics_temp["mAP"])
    
    map_50_95 = np.mean(map_scores)
    
    # Compile all metrics
    results = {
        "test_samples": len(test_dataset),
        "score_threshold": score_threshold,
        "metrics_at_iou_0.5": {
            "precision": metrics_05["precision"],
            "recall": metrics_05["recall"],
            "f1": metrics_05["f1"],
            "mAP": metrics_05["mAP"],
            "tp": metrics_05["tp"],
            "fp": metrics_05["fp"],
            "fn": metrics_05["fn"],
            "tn": metrics_05["tn"],
            "total_predictions": metrics_05["total_predictions"],
        },
        "mAP@0.5:0.95": map_50_95,
    }
    
    # Save metrics to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = output_dir / f"test_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Test Results (IoU=0.5, score_threshold={score_threshold})")
    print(f"{'='*60}")
    print(f"Precision: {metrics_05['precision']:.4f}")
    print(f"Recall:    {metrics_05['recall']:.4f}")
    print(f"F1 Score:  {metrics_05['f1']:.4f}")
    print(f"mAP@0.5:   {metrics_05['mAP']:.4f}")
    print(f"mAP@0.5:0.95: {map_50_95:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics_05['tp']}")
    print(f"  FP: {metrics_05['fp']}")
    print(f"  FN: {metrics_05['fn']}")
    print(f"  TN: {metrics_05['tn']}")
    print(f"\nTotal predictions: {metrics_05['total_predictions']}")
    print(f"{'='*60}")
    
    # Visualize a subset of samples
    print(f"\nGenerating visualizations for {visualize_samples} samples...")
    viz_indices = np.linspace(0, len(test_dataset) - 1, min(visualize_samples, len(test_dataset)), dtype=int)
    
    for idx in tqdm(viz_indices, desc="Visualizing"):
        sample_name = all_file_paths[idx].stem
        
        # Filter predictions by score threshold
        keep = all_pred_scores[idx] >= score_threshold
        pred_boxes_filtered = all_pred_boxes[idx][keep]
        pred_scores_filtered = all_pred_scores[idx][keep]
        pred_labels_filtered = all_pred_labels[idx][keep]
        
        viz_path = viz_dir / f"{sample_name}_inference.png"
        visualize_detection(
            all_images[idx],
            pred_boxes_filtered,
            pred_scores_filtered,
            pred_labels_filtered,
            all_gt_boxes[idx],
            all_gt_labels[idx],
            viz_path,
            sample_name
        )
    
    print(f"\nResults saved:")
    print(f"  Metrics: {metrics_path}")
    print(f"  Visualizations: {viz_dir}")
    
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference with metrics for OCT detection")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (.pth)"
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=Path("/home/suraj/Git/RCNN-OCT/splits.json"),
        help="Path to splits.json file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inference"),
        help="Directory to save results (default: inference/)"
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Score threshold for predictions (default: 0.5)"
    )
    parser.add_argument(
        "--visualize-samples",
        type=int,
        default=10,
        help="Number of samples to visualize (default: 10)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test dataset
    print(f"Loading test dataset from {args.splits_file}...")
    test_dataset, label_mapping = create_datasets_from_splits(
        args.splits_file,
        split_names=("test",),
        filter_empty_ratio=0.0,
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Label mapping: {label_mapping}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=device)
    
    # Run batch inference
    batch_inference(
        model=model,
        test_dataset=test_dataset,
        device=device,
        score_threshold=args.score_threshold,
        output_dir=args.output_dir,
        visualize_samples=args.visualize_samples,
    )


if __name__ == "__main__":
    main()
