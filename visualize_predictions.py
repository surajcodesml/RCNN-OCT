"""
Visualize model predictions vs ground truth with confusion analysis.

This script loads evaluation results, visualizes predictions overlaid with ground truth,
and generates a confusion matrix and detailed analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import seaborn as sns

from inference import load_model


def load_coco_annotations(json_path: Path) -> Tuple[Dict, Dict, Dict]:
    """Load COCO format annotations."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images_dict = {img['id']: img for img in data['images']}
    
    annotations_dict = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_dict:
            annotations_dict[image_id] = []
        annotations_dict[image_id].append(ann)
    
    categories_dict = {cat['id']: cat for cat in data['categories']}
    
    return images_dict, annotations_dict, categories_dict


def extract_image_name(file_path: str) -> str:
    """Extract image filename from annotation path."""
    parts = file_path.replace('\\\\', '/').split('/')
    filename = parts[-1]
    
    if '-' in filename:
        filename_parts = filename.split('-', 1)
        if len(filename_parts) > 1:
            actual_name = filename_parts[1]
            return actual_name
    
    return filename


def load_image_tensor(image_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
    """Load image and convert to model input format."""
    image = Image.open(image_path).convert('L')
    image_np = np.array(image)
    
    image_tensor = torch.from_numpy(image_np.astype(np.float32))
    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.repeat(3, 1, 1)
    image_tensor = image_tensor / 255.0
    
    return image_tensor, image_np


def run_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    score_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on image."""
    model.eval()
    with torch.no_grad():
        outputs = model([image_tensor.to(device)])
    
    output = outputs[0]
    scores = output['scores'].cpu().numpy()
    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    
    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    return boxes, scores, labels


def coco_bbox_to_xyxy(bbox: List[float]) -> np.ndarray:
    """Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h])


def visualize_detection(
    image_np: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    categories_dict: Dict,
    save_path: Path,
    title: str = ""
) -> None:
    """Visualize predictions and ground truth on image."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(image_np, cmap='gray')
    
    # Define colors for each class
    colors = {0: 'red', 1: 'blue', 2: 'yellow'}
    
    # Draw ground truth boxes (dashed lines)
    for gt_box, gt_label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = gt_box
        width = x2 - x1
        height = y2 - y1
        color = colors.get(gt_label, 'green')
        class_name = categories_dict.get(gt_label, {}).get('name', f'class_{gt_label}')
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none',
            linestyle='--', label=f'GT: {class_name}'
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'GT: {class_name}', 
                color=color, fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Draw predicted boxes (solid lines)
    for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = pred_box
        width = x2 - x1
        height = y2 - y1
        color = colors.get(pred_label, 'cyan')
        class_name = categories_dict.get(pred_label, {}).get('name', f'class_{pred_label}')
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none',
            linestyle='-', label=f'Pred: {class_name}'
        )
        ax.add_patch(rect)
        ax.text(x2, y2 + 15, f'Pred: {class_name} ({pred_score:.2f})', 
                color=color, fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_title(title, fontsize=14, weight='bold')
    ax.axis('off')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='w', markerfacecolor='none', marker='s', 
               markersize=10, markeredgewidth=2, markeredgecolor='gray', 
               linestyle='--', label='Ground Truth (dashed)'),
        Line2D([0], [0], color='w', markerfacecolor='none', marker='s', 
               markersize=10, markeredgewidth=2, markeredgecolor='gray', 
               linestyle='-', label='Prediction (solid)')
    ]
    for class_id, class_info in categories_dict.items():
        color = colors.get(class_id, 'green')
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=3, label=class_info['name'])
        )
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_confusion_matrix(
    pred_boxes_all: List[np.ndarray],
    pred_labels_all: List[np.ndarray],
    gt_boxes_all: List[np.ndarray],
    gt_labels_all: List[np.ndarray],
    num_classes: int = 3,
    iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Compute confusion matrix for object detection.
    
    Rows are ground truth classes, columns are predicted classes.
    An additional column is added for false negatives (missed detections).
    """
    # Initialize confusion matrix (GT classes x Pred classes + 1 for FN)
    confusion = np.zeros((num_classes, num_classes + 1), dtype=int)
    
    for pred_boxes, pred_labels, gt_boxes, gt_labels in zip(
        pred_boxes_all, pred_labels_all, gt_boxes_all, gt_labels_all
    ):
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        
        # For each prediction, find best matching GT
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_idx = -1
            best_gt_label = -1
            
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if not gt_matched[gt_idx]:
                    iou = bbox_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gt_idx
                        best_gt_label = gt_label
            
            if best_iou >= iou_threshold and best_idx >= 0:
                # Matched prediction
                confusion[best_gt_label, pred_label] += 1
                gt_matched[best_idx] = True
            # Note: unmatched predictions (false positives) are not added to confusion matrix
            # as they don't have a ground truth class
        
        # Add false negatives (unmatched ground truths)
        for gt_idx, gt_label in enumerate(gt_labels):
            if not gt_matched[gt_idx]:
                confusion[gt_label, num_classes] += 1  # Last column is for FN
    
    return confusion


def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate IoU between two boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def plot_confusion_matrix(
    confusion: np.ndarray,
    categories_dict: Dict,
    save_path: Path
) -> None:
    """Plot confusion matrix heatmap."""
    num_classes = len(categories_dict)
    
    # Create labels
    class_names = [categories_dict[i]['name'] for i in range(num_classes)]
    col_labels = class_names + ['Missed (FN)']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        confusion, annot=True, fmt='d', cmap='Blues',
        xticklabels=col_labels, yticklabels=class_names,
        ax=ax, cbar_kws={'label': 'Count'}
    )
    
    ax.set_xlabel('Predicted Class', fontsize=12, weight='bold')
    ax.set_ylabel('Ground Truth Class', fontsize=12, weight='bold')
    ax.set_title('Confusion Matrix\n(Rows: GT, Columns: Predicted + Missed)', 
                 fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize predictions and create confusion matrix')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--annotations', type=str, required=True, help='Path to annotations JSON')
    parser.add_argument('--images-dir', type=str, required=True, help='Path to images directory')
    parser.add_argument('--score-threshold', type=float, default=0.5, help='Score threshold')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--num-samples', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='visualization_output', help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(Path(args.checkpoint), num_classes=args.num_classes, device=device)
    
    # Load annotations
    print(f'Loading annotations...')
    images_dict, annotations_dict, categories_dict = load_coco_annotations(Path(args.annotations))
    
    print(f'Found {len(images_dict)} images')
    print(f'Categories: {categories_dict}')
    
    images_dir = Path(args.images_dir)
    
    # Collect all predictions and ground truths
    all_pred_boxes = []
    all_pred_labels = []
    all_pred_scores = []
    all_gt_boxes = []
    all_gt_labels = []
    all_image_ids = []
    
    print('Running inference on all images...')
    for image_id, image_info in tqdm(images_dict.items()):
        image_name = extract_image_name(image_info['file_name'])
        
        image_path = None
        for img_file in images_dir.iterdir():
            if img_file.name.endswith(image_name):
                image_path = img_file
                break
        
        if image_path is None:
            continue
        
        image_tensor, image_np = load_image_tensor(image_path)
        pred_boxes, pred_scores, pred_labels = run_inference(
            model, image_tensor, device, args.score_threshold
        )
        
        gt_annotations = annotations_dict.get(image_id, [])
        gt_boxes = []
        gt_labels = []
        
        for ann in gt_annotations:
            gt_boxes.append(coco_bbox_to_xyxy(ann['bbox']))
            gt_labels.append(ann['category_id'])
        
        gt_boxes = np.array(gt_boxes) if len(gt_boxes) > 0 else np.zeros((0, 4))
        gt_labels = np.array(gt_labels) if len(gt_labels) > 0 else np.zeros((0,), dtype=int)
        
        all_pred_boxes.append(pred_boxes)
        all_pred_labels.append(pred_labels)
        all_pred_scores.append(pred_scores)
        all_gt_boxes.append(gt_boxes)
        all_gt_labels.append(gt_labels)
        all_image_ids.append(image_id)
    
    # Generate confusion matrix
    print('\\nGenerating confusion matrix...')
    print('\nGenerating confusion matrix...')
    confusion = compute_confusion_matrix(
        all_pred_boxes, all_pred_labels, all_gt_boxes, all_gt_labels,
        num_classes=args.num_classes
    )
    
    print('\nConfusion Matrix:')
    print('Rows: Ground Truth, Columns: Predicted + Missed')
    class_names = [categories_dict[i]['name'] for i in range(args.num_classes)]
    col_labels = class_names + ['Missed']
    header = f"{'GT/Pred':<15} " + " ".join(f"{label:<12}" for label in col_labels)
    print(header)
    for i, row_label in enumerate(class_names):
        row_str = f"{row_label:<15} " + " ".join(f"{val:<12}" for val in confusion[i])
        print(row_str)
    
    plot_confusion_matrix(confusion, categories_dict, output_dir / 'confusion_matrix.png')
    print(f'Saved confusion matrix to {output_dir / "confusion_matrix.png"}')
    
    # Visualize sample images
    print(f'\nVisualizing {args.num_samples} sample images...')
    
    # Select diverse samples
    sample_indices = random.sample(range(len(all_image_ids)), min(args.num_samples, len(all_image_ids)))
    
    for idx, sample_idx in enumerate(tqdm(sample_indices)):
        image_id = all_image_ids[sample_idx]
        image_info = images_dict[image_id]
        image_name = extract_image_name(image_info['file_name'])
        
        image_path = None
        for img_file in images_dir.iterdir():
            if img_file.name.endswith(image_name):
                image_path = img_file
                break
        
        if image_path is None:
            continue
        
        _, image_np = load_image_tensor(image_path)
        
        pred_boxes = all_pred_boxes[sample_idx]
        pred_labels = all_pred_labels[sample_idx]
        pred_scores = all_pred_scores[sample_idx]
        gt_boxes = all_gt_boxes[sample_idx]
        gt_labels = all_gt_labels[sample_idx]
        
        title = f'Image {idx+1}: {image_name}\\nGT: {len(gt_boxes)} boxes, Pred: {len(pred_boxes)} boxes'
        
        visualize_detection(
            image_np, pred_boxes, pred_labels, pred_scores,
            gt_boxes, gt_labels, categories_dict,
            output_dir / f'sample_{idx+1:03d}_{image_name}',
            title=title
        )
    
    print(f'\\nVisualization complete! Results saved to {output_dir}')
    
    # Save summary statistics
    summary = {
        'total_images': len(all_image_ids),
        'total_ground_truths': sum(len(boxes) for boxes in all_gt_boxes),
        'total_predictions': sum(len(boxes) for boxes in all_pred_boxes),
        'confusion_matrix': confusion.tolist(),
        'class_names': class_names
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'\\nSummary:')
    print(f'  Total images: {summary["total_images"]}')
    print(f'  Total ground truths: {summary["total_ground_truths"]}')
    print(f'  Total predictions: {summary["total_predictions"]}')


if __name__ == '__main__':
    main()
