"""
Visualize RCNN predictions vs ground truth with CORRECTED label mapping.

Applies same label remapping as evaluation:
- Model predictions: class 2 -> SCR (eval class 0), class 1 -> fovea (eval class 1)  
- Ground truth: "maybe" (class 2) -> SCR (class 0)
- Uses x-axis for visualization clarity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

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
    """Extract image filename."""
    parts = file_path.replace('\\\\', '/').split('/')
    filename = parts[-1]
    
    if '-' in filename:
        filename_parts = filename.split('-', 1)
        if len(filename_parts) > 1:
            return filename_parts[1]
    
    return filename


def load_image_tensor(image_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
    """Load image for model."""
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
    """Run inference."""
    model.eval()
    with torch.no_grad():
        outputs = model([image_tensor.to(device)])
    
    output = outputs[0]
    scores = output['scores'].cpu().numpy()
    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    
    keep = scores >= score_threshold
    return boxes[keep], scores[keep], labels[keep]


def coco_bbox_to_xyxy(bbox: List[float]) -> np.ndarray:
    """Convert COCO bbox to xyxy."""
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h])


def remap_labels(labels: np.ndarray, label_mapping: Dict[int, int]) -> np.ndarray:
    """Remap labels according to mapping."""
    if len(labels) == 0:
        return labels
    return np.array([label_mapping.get(int(label), int(label)) for label in labels])


def visualize_detection(
    image_np: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    save_path: Path,
    title: str = ""
) -> None:
    """Visualize with CORRECTED color mapping (eval classes: 0=SCR red, 1=fovea blue)."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(image_np, cmap='gray')
    
    # Colors for EVAL classes (after remapping): 0=SCR, 1=fovea
    colors = {0: 'red', 1: 'blue'}
    class_names = {0: 'SCR', 1: 'fovea'}
    
    # Draw ground truth boxes (dashed)
    for gt_box, gt_label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = gt_box
        color = colors.get(gt_label, 'green')
        class_name = class_names.get(gt_label, f'class_{gt_label}')
        
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'GT: {class_name}', 
                color=color, fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Draw predictions (solid)
    for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = pred_box
        color = colors.get(pred_label, 'cyan')
        class_name = class_names.get(pred_label, f'class_{pred_label}')
        
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none',
            linestyle='-'
        )
        ax.add_patch(rect)
        ax.text(x2, y2 + 15, f'Pred: {class_name} ({pred_score:.2f})', 
                color=color, fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_title(title, fontsize=14, weight='bold')
    ax.axis('off')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='w', markerfacecolor='none', marker='s', 
               markersize=10, markeredgewidth=2, markeredgecolor='gray', 
               linestyle='--', label='Ground Truth (dashed)'),
        Line2D([0], [0], color='w', markerfacecolor='none', marker='s', 
               markersize=10, markeredgewidth=2, markeredgecolor='gray', 
               linestyle='-', label='Prediction (solid)'),
        Line2D([0], [0], color='red', linewidth=3, label='SCR'),
        Line2D([0], [0], color='blue', linewidth=3, label='fovea')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize RCNN predictions with correct label mapping')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--annotations', type=str, required=True)
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--score-threshold', type=float, default=0.5)
    parser.add_argument('--num-samples', type=int, default=30)
    parser.add_argument('--output-dir', type=str, default='visualization_corrected')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model with 3 classes
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(Path(args.checkpoint), num_classes=3, device=device)
    
    # Load model label mapping
    checkpoint = torch.load(Path(args.checkpoint), map_location='cpu')
    model_label_mapping = checkpoint.get('label_mapping', {})
    print(f'Model label mapping: {model_label_mapping}')
    
    # Create prediction remapping: model class -> eval class
    # Training: {0: fovea, 1: SCR} -> Model outputs: {1: fovea, 2: SCR}
    # Eval: {0: SCR, 1: fovea}
    pred_remap = {}
    if model_label_mapping:
        for train_label, model_class in model_label_mapping.items():
            if int(train_label) == 0:  # fovea in training
                pred_remap[int(model_class)] = 1  # fovea in eval
            elif int(train_label) == 1:  # SCR in training
                pred_remap[int(model_class)] = 0  # SCR in eval
    
    print(f'Prediction remapping: {pred_remap} (model class -> eval class)')
    
    # Load annotations
    print(f'Loading annotations...')
    images_dict, annotations_dict, categories_dict = load_coco_annotations(Path(args.annotations))
    
    images_dir = Path(args.images_dir)
    all_image_ids = list(images_dict.keys())
    
    # Sample random images
    sample_ids = random.sample(all_image_ids, min(args.num_samples, len(all_image_ids)))
    
    print(f'Generating {len(sample_ids)} visualizations...')
    for idx, image_id in enumerate(tqdm(sample_ids)):
        image_info = images_dict[image_id]
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
        
        # Remap prediction labels
        pred_labels = remap_labels(pred_labels, pred_remap)
        
        # Get ground truth and remap
        gt_annotations = annotations_dict.get(image_id, [])
        gt_boxes = []
        gt_labels = []
        
        for ann in gt_annotations:
            gt_boxes.append(coco_bbox_to_xyxy(ann['bbox']))
            # Remap GT: COCO 1(fovea)->eval 1, COCO 0(SCR)->eval 0, COCO 2(maybe)->eval 0
            coco_label = ann['category_id']
            if coco_label == 1:  # fovea
                gt_labels.append(1)
            else:  # SCR or maybe
                gt_labels.append(0)
        
        gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))
        gt_labels = np.array(gt_labels) if gt_labels else np.zeros((0,), dtype=int)
        
        title = f'Image {idx+1}: {image_name}\\nGT: {len(gt_boxes)} boxes (after maybe->SCR), Pred: {len(pred_boxes)} boxes'
        
        visualize_detection(
            image_np, pred_boxes, pred_labels, pred_scores,
            gt_boxes, gt_labels,
            output_dir / f'sample_{idx+1:03d}_{image_name}',
            title=title
        )
    
    print(f'\\nVisualizations saved to {output_dir}/')
    print('Colors: RED = SCR, BLUE = fovea')
    print('Dashed lines = Ground truth, Solid lines = Predictions')


if __name__ == '__main__':
    main()
