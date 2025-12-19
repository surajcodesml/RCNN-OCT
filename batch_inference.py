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
import yaml
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
    all_pred_labels: List[np.ndarray],
    all_gt_boxes: List[np.ndarray],
    all_gt_labels: List[np.ndarray],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Compute detection metrics at a specific IoU threshold.
    
    Returns:
        Dictionary with precision, recall, f1, mAP (overall and per-class), TP, FP, FN, TN
    """
    # Compute TP, FP, FN at IoU threshold (overall)
    tp = fp = fn = tn = 0
    total_preds = 0
    
    # Per-class metrics
    class_tp = {1: 0, 2: 0}  # Fovea: 1, SCR: 2
    class_fp = {1: 0, 2: 0}
    class_fn = {1: 0, 2: 0}
    
    for pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels in zip(
        all_pred_boxes, all_pred_scores, all_pred_labels, all_gt_boxes, all_gt_labels
    ):
        # Filter by score threshold
        keep = pred_scores >= score_threshold
        pred_boxes_filtered = pred_boxes[keep]
        pred_labels_filtered = pred_labels[keep]
        
        total_preds += len(pred_boxes_filtered)
        
        matched_gt = set()
        for pred_box, pred_label in zip(pred_boxes_filtered, pred_labels_filtered):
            pred_class = int(pred_label)
            
            if len(gt_boxes) == 0:
                fp += 1
                if pred_class in class_fp:
                    class_fp[pred_class] += 1
                continue
            
            # Find best matching GT box with same class
            best_iou = 0.0
            best_idx = -1
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if int(gt_label) == pred_class and gt_idx not in matched_gt:
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gt_idx
            
            if best_iou >= iou_threshold and best_idx >= 0:
                tp += 1
                matched_gt.add(best_idx)
                if pred_class in class_tp:
                    class_tp[pred_class] += 1
            else:
                fp += 1
                if pred_class in class_fp:
                    class_fp[pred_class] += 1
        
        # Count unmatched GT boxes (FN) per class
        for gt_idx, gt_label in enumerate(gt_labels):
            if gt_idx not in matched_gt:
                fn += 1
                gt_class = int(gt_label)
                if gt_class in class_fn:
                    class_fn[gt_class] += 1
        
        # TN: Images without GT boxes where we correctly predict nothing
        if len(gt_boxes) == 0 and len(pred_boxes_filtered) == 0:
            tn += 1
    
    # Compute mAP at this IoU threshold (overall and per-class)
    # Organize predictions by class
    class_predictions = {1: [], 2: []}  # class_id: [(score, pred_box, image_idx)]
    
    for img_idx, (pred_boxes, pred_scores, pred_labels) in enumerate(zip(all_pred_boxes, all_pred_scores, all_pred_labels)):
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            class_id = int(label)
            if class_id in class_predictions:
                class_predictions[class_id].append((score, box, img_idx))
    
    # Compute per-class AP
    class_ap = {}
    for class_id in [1, 2]:  # Fovea: 1, SCR: 2
        predictions = class_predictions[class_id]
        predictions.sort(key=lambda x: x[0], reverse=True)
        
        # Count GT boxes for this class
        total_gt_class = sum(
            np.sum(gt_labels == class_id) 
            for gt_labels in all_gt_labels
        )
        
        if total_gt_class == 0:
            class_ap[class_id] = 0.0
            continue
        
        # Track matched GT boxes per image
        matched_gts_per_image = [set() for _ in all_gt_boxes]
        
        tp_curve = 0
        fp_curve = 0
        precisions = []
        recalls = []
        
        for score, pred_box, img_idx in predictions:
            gt_boxes = all_gt_boxes[img_idx]
            gt_labels = all_gt_labels[img_idx]
            
            # Filter GT boxes by class
            class_mask = gt_labels == class_id
            class_gt_boxes = gt_boxes[class_mask]
            
            if len(class_gt_boxes) == 0:
                fp_curve += 1
            else:
                # Map local indices to original indices
                class_indices = np.where(class_mask)[0]
                
                # Find best matching GT box
                best_iou = 0.0
                best_local_idx = -1
                for local_idx, gt_box in enumerate(class_gt_boxes):
                    original_idx = class_indices[local_idx]
                    if original_idx not in matched_gts_per_image[img_idx]:
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_local_idx = local_idx
                
                if best_iou >= iou_threshold and best_local_idx >= 0:
                    tp_curve += 1
                    matched_gts_per_image[img_idx].add(class_indices[best_local_idx])
                else:
                    fp_curve += 1
            
            precision = tp_curve / (tp_curve + fp_curve) if (tp_curve + fp_curve) > 0 else 0.0
            recall = tp_curve / total_gt_class if total_gt_class > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)
        
        # Compute AP for this class
        if len(precisions) > 0:
            precisions = np.array(precisions)
            recalls = np.array(recalls)
            class_ap[class_id] = compute_ap(precisions, recalls)
        else:
            class_ap[class_id] = 0.0
    
    # Overall mAP is mean of per-class APs
    mAP = np.mean(list(class_ap.values()))
    
    # Compute per-class precision, recall, F1
    class_metrics = {}
    for class_id in [1, 2]:
        c_tp = class_tp[class_id]
        c_fp = class_fp[class_id]
        c_fn = class_fn[class_id]
        
        c_precision = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_recall = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = 2 * c_precision * c_recall / (c_precision + c_recall) if (c_precision + c_recall) > 0 else 0.0
        
        class_name = "Fovea" if class_id == 1 else "SCR"
        class_metrics[class_name] = {
            "precision": c_precision,
            "recall": c_recall,
            "f1": c_f1,
            "ap": class_ap[class_id],
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
        }
    
    # Final overall metrics
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
        "per_class": class_metrics,
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
    """Visualize predictions vs ground truth with color-coded labels.
    
    Color coding:
        - Label 1 (Fovea): Green
        - Label 2 (SCR): Red
    """
    # Label names and colors
    label_names = {1: "Fovea", 2: "SCR"}
    label_colors = {1: "green", 2: "red"}
    
    # Convert image to numpy
    if image.dim() == 3 and image.shape[0] == 3:
        image_np = image[0].cpu().numpy()
    else:
        image_np = image.squeeze().cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot ground truth
    ax1.imshow(image_np, cmap="gray")
    ax1.set_title(f"Ground Truth - {sample_name}", fontsize=14, fontweight="bold")
    ax1.axis("off")
    
    for box, label in zip(gt_boxes, gt_labels):
        label_int = int(label)
        color = label_colors.get(label_int, "yellow")
        label_text = label_names.get(label_int, f"Class {label_int}")
        
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=color, facecolor="none", linestyle="-"
        )
        ax1.add_patch(rect)
        
        # Add label with background
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7, edgecolor="none")
        ax1.text(
            x1, y1 - 8, f"GT: {label_text}", 
            color="white", fontsize=10, weight="bold", 
            bbox=bbox_props, verticalalignment="top"
        )
    
    # Plot predictions
    ax2.imshow(image_np, cmap="gray")
    ax2.set_title(f"Predictions - {sample_name}", fontsize=14, fontweight="bold")
    ax2.axis("off")
    
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        label_int = int(label)
        color = label_colors.get(label_int, "yellow")
        label_text = label_names.get(label_int, f"Class {label_int}")
        
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=color, facecolor="none", linestyle="-"
        )
        ax2.add_patch(rect)
        
        # Add label with background
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7, edgecolor="none")
        ax2.text(
            x1, y1 - 8, f"{label_text}: {score:.2f}", 
            color="white", fontsize=10, weight="bold", 
            bbox=bbox_props, verticalalignment="top"
        )
    
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
        all_pred_boxes, all_pred_scores, all_pred_labels, all_gt_boxes, all_gt_labels,
        iou_threshold=0.5, score_threshold=score_threshold
    )
    
    # Compute mAP@0.5:0.95 (average over IoU thresholds 0.5, 0.55, ..., 0.95)
    # Overall and per-class
    map_scores_overall = []
    map_scores_fovea = []
    map_scores_scr = []
    
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        metrics_temp = compute_metrics_at_iou(
            all_pred_boxes, all_pred_scores, all_pred_labels, all_gt_boxes, all_gt_labels,
            iou_threshold=iou_thresh, score_threshold=score_threshold
        )
        map_scores_overall.append(metrics_temp["mAP"])
        map_scores_fovea.append(metrics_temp["per_class"]["Fovea"]["ap"])
        map_scores_scr.append(metrics_temp["per_class"]["SCR"]["ap"])
    
    map_50_95_overall = np.mean(map_scores_overall)
    map_50_95_fovea = np.mean(map_scores_fovea)
    map_50_95_scr = np.mean(map_scores_scr)
    
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
            "per_class": metrics_05["per_class"],
        },
        "mAP@0.5:0.95": {
            "overall": map_50_95_overall,
            "Fovea": map_50_95_fovea,
            "SCR": map_50_95_scr,
        },
    }
    
    # Save metrics to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = output_dir / f"inference_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Test Results (IoU=0.5, score_threshold={score_threshold})")
    print(f"{'='*60}")
    print(f"Overall Metrics:")
    print(f"  Precision: {metrics_05['precision']:.4f}")
    print(f"  Recall:    {metrics_05['recall']:.4f}")
    print(f"  F1 Score:  {metrics_05['f1']:.4f}")
    print(f"  mAP@0.5:   {metrics_05['mAP']:.4f}")
    print(f"  mAP@0.5:0.95: {map_50_95_overall:.4f}")
    print(f"\nPer-Class Metrics (IoU=0.5):")
    for class_name, class_metrics in metrics_05["per_class"].items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1 Score:  {class_metrics['f1']:.4f}")
        print(f"    AP@0.5:    {class_metrics['ap']:.4f}")
    print(f"\nPer-Class mAP@0.5:0.95:")
    print(f"  Fovea: {map_50_95_fovea:.4f}")
    print(f"  SCR:   {map_50_95_scr:.4f}")
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
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (.pth)"
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=None,
        help="Path to splits.json file (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save results (overrides config)"
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Score threshold for predictions (overrides config)"
    )
    parser.add_argument(
        "--visualize-samples",
        type=int,
        default=None,
        help="Number of samples to visualize (overrides config)"
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Override config with command line arguments
    splits_file = args.splits_file if args.splits_file else Path(config["data"]["splits_file"])
    output_dir = args.output_dir if args.output_dir else Path(config["output"]["inference_dir"])
    score_threshold = args.score_threshold if args.score_threshold is not None else config["inference"]["score_threshold"]
    visualize_samples = args.visualize_samples if args.visualize_samples is not None else config["inference"]["visualize_samples"]
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and config["device"]["cuda"] else "cpu")
    print(f"Using device: {device}")
    
    # Verify splits file exists
    if not splits_file.exists():
        raise FileNotFoundError(
            f"Splits file not found: {splits_file}\n"
            f"Please run 'python split.py' first to generate the splits."
        )
    
    # Load test dataset
    print(f"Loading test dataset from {splits_file}...")
    test_dataset, label_mapping = create_datasets_from_splits(
        splits_file,
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
        score_threshold=score_threshold,
        output_dir=output_dir,
        visualize_samples=visualize_samples,
    )


if __name__ == "__main__":
    main()
