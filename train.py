"""Training and evaluation script for Faster R-CNN on OCT B-scans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from dataset import create_datasets_from_splits, detection_collate_fn
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on OCT data")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=None,
        help="Path to splits.json file (overrides config)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for checkpoints (overrides config)"
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.05,
) -> Dict[str, float]:
    """Evaluate model and compute precision, recall, F1, and mAP."""
    model.eval()
    
    # Collect all predictions and ground truths
    all_pred_boxes = []
    all_pred_scores = []
    all_gt_boxes = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()
                gt_boxes = target["boxes"].cpu().numpy()
                
                all_pred_boxes.append(pred_boxes)
                all_pred_scores.append(pred_scores)
                all_gt_boxes.append(gt_boxes)
    
    # Compute metrics at IoU=0.5
    tp = fp = fn = 0
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
            
            if ious[best_idx] >= 0.5 and best_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        
        fn += len(gt_boxes) - len(matched_gt)
    
    # Compute mAP@0.5
    # Sort all predictions by score
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
            
            if ious[best_idx] >= 0.5 and best_idx not in matched_gts[gt_idx]:
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
        "total_predictions": total_preds,
    }


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    loss_accum = {"loss": 0.0, "loss_classifier": 0.0, "loss_box_reg": 0.0, "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0}
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Accumulate all loss components
        for key in loss_accum:
            loss_accum[key] += loss_dict.get(key, torch.tensor(0.0, device=device)).item()
        num_batches += 1
        
        # Update progress bar with current loss
        progress_bar.set_postfix({"loss": f"{losses.item():.4f}"})

    for key in loss_accum:
        loss_accum[key] /= max(num_batches, 1)
    
    # Compute total loss as sum of all components
    loss_accum["loss"] = sum(loss_accum[k] for k in loss_accum if k != "loss")
    
    return loss_accum

#Function to save training results to JSON
def save_training_results(
    output_dir: Path,
    hyperparameters: Dict,
    history: List[Dict],
    best_metrics: Dict,
) -> None:
    """Save hyperparameters, training history, and best metrics to JSON file."""
    results = {
        "hyperparameters": hyperparameters,
        "training_history": history,
        "best_metrics": best_metrics,
    }
    
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Training results saved to {results_path}")

def main() -> None:
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Override config with command line arguments
    splits_file = args.splits_file if args.splits_file else Path(config["data"]["splits_file"])
    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else config["training"]["batch_size"]
    output_dir = args.output_dir if args.output_dir else Path(config["output"]["checkpoints_dir"])
    
    # Training hyperparameters from config
    learning_rate = config["training"]["learning_rate"]
    momentum = config["training"]["momentum"]
    weight_decay = config["training"]["weight_decay"]
    lr_step_size = config["training"]["lr_scheduler_step_size"]
    lr_gamma = config["training"]["lr_scheduler_gamma"]
    num_workers = config["training"]["num_workers"]
    pin_memory = config["training"]["pin_memory"]
    patience = config["training"]["patience"]
    filter_empty = config["training"]["filter_empty_ratio"]
    score_threshold = config["training"]["score_threshold"]
    seed = config["splitting"]["seed"]
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and config["device"]["cuda"] else "cpu")
    print(f"Using device: {device}")

    # Load datasets from splits file
    if not splits_file.exists():
        raise FileNotFoundError(
            f"Splits file not found: {splits_file}\n"
            f"Please run 'python split.py' first to generate the splits."
        )
    
    print(f"Loading datasets from {splits_file}...")
    train_dataset, val_dataset, label_mapping = create_datasets_from_splits(
        splits_file,
        split_names=("train", "val"),
        filter_empty_ratio=filter_empty,
        seed=seed
    )
    
    num_classes = len(label_mapping) + 1 if label_mapping else 2

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Label mapping: {label_mapping}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=detection_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=detection_collate_fn,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory
    )

    print("Building model...")
    model = build_model(num_classes=num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_path = output_dir / "best_model.pth"
    epochs_without_improvement = 0
    
    hyperparameters = {
        "config_file": str(args.config),
        "splits_file": str(splits_file),
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "score_threshold": score_threshold,
        "filter_empty_ratio": filter_empty,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "lr_scheduler_step_size": lr_step_size,
        "lr_scheduler_gamma": lr_gamma,
        "patience": patience,
        "num_workers": num_workers,
        "num_classes": num_classes,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "label_mapping": label_mapping,
        "device": str(device),
    }
    training_history = []

    print(f"\nStarting training for {epochs} epochs...\n")
    for epoch in range(1, epochs + 1):
        loss_dict = train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()

        val_metrics = evaluate(model, val_loader, device, score_threshold=score_threshold)

        epoch_results = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": loss_dict["loss"],
            "train_loss_classifier": loss_dict["loss_classifier"],
            "train_loss_box_reg": loss_dict["loss_box_reg"],
            "train_loss_objectness": loss_dict["loss_objectness"],
            "train_loss_rpn_box_reg": loss_dict["loss_rpn_box_reg"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_mAP": val_metrics["mAP"],
            "val_tp": val_metrics["tp"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
            "val_total_predictions": val_metrics["total_predictions"],
        }
        training_history.append(epoch_results)
        
        print(
            f"Epoch {epoch}/{epochs}: "  
            f"loss={loss_dict['loss']:.4f}, "
            f"cls={loss_dict['loss_classifier']:.4f}, "
            f"box={loss_dict['loss_box_reg']:.4f}, "
            f"obj={loss_dict['loss_objectness']:.4f}, "
            f"rpn_box={loss_dict['loss_rpn_box_reg']:.4f} | "  
            f"P={val_metrics['precision']:.4f}, "
            f"R={val_metrics['recall']:.4f}, "
            f"F1={val_metrics['f1']:.4f}, "
            f"mAP={val_metrics['mAP']:.4f} "
            f"(TP={val_metrics['tp']}, FP={val_metrics['fp']}, FN={val_metrics['fn']}, Preds={val_metrics['total_predictions']})" 
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({"model_state": model.state_dict(), "label_mapping": label_mapping}, best_path)
            print(f"Saved new best model to {best_path} with F1={best_f1:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")
            
            # Stop training if patience exceeded
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best F1: {best_f1:.4f} achieved at epoch {epoch - patience}")
                break
        
    best_metrics = {
        "best_f1": best_f1,
        "best_epoch": max(training_history, key=lambda x: x["val_f1"])["epoch"],
        "final_train_loss": training_history[-1]["train_loss"],
        "final_val_precision": training_history[-1]["val_precision"],
        "final_val_recall": training_history[-1]["val_recall"],
        "final_val_f1": training_history[-1]["val_f1"],
        "final_val_mAP": training_history[-1]["val_mAP"],
    }

    save_training_results(output_dir, hyperparameters, training_history, best_metrics)
        

if __name__ == "__main__":
    main()
