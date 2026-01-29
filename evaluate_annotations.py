"""
Evaluate trained RCNN model on annotated test dataset.

This script loads a test dataset with images and bounding box annotations in COCO format,
runs inference with the trained model, and computes detection metrics including:
- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

IMPORTANT: 
- Ground truth "maybe" labels are remapped to SCR (class 0) since model was trained on fovea/SCR only
- Uses x-axis IoU (horizontal overlap) instead of 2D IoU since y-axis differences don't matter for OCT B-scans
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from inference import load_model


def load_coco_annotations(json_path: Path) -> Tuple[Dict, Dict, Dict]:
    """
    Load COCO format annotations from JSON file.
    
    Returns:
        images_dict: Mapping from image_id to image metadata
        annotations_dict: Mapping from image_id to list of annotations
        categories_dict: Mapping from category_id to category info
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images_dict = {img['id']: img for img in data['images']}
    
    # Group annotations by image_id
    annotations_dict = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_dict:
            annotations_dict[image_id] = []
        annotations_dict[image_id].append(ann)
    
    categories_dict = {cat['id']: cat for cat in data['categories']}
    
    return images_dict, annotations_dict, categories_dict


def extract_image_name(file_path: str) -> str:
    """
    Extract just the image filename from a path that may have incorrect directory structure.
    
    Example: "../../home/mbtasepta/.../3a0e5b02-22_R_5_bscan_020.png" -> "22_R_5_bscan_020.png"
    """
    parts = file_path.replace('\\\\', '/').split('/')
    filename = parts[-1]
    
    if '-' in filename:
        filename_parts = filename.split('-', 1)
        if len(filename_parts) > 1:
            actual_name = filename_parts[1]
            return actual_name
    
    return filename


def load_image(image_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load image and convert to model input format.
    
    Returns:
        image_tensor: Tensor in format [3, H, W] normalized to [0, 1]
        image_np: Original image as numpy array for visualization
    """
    image = Image.open(image_path).convert('L')
    image_np = np.array(image)
    
    image_tensor = torch.from_numpy(image_np.astype(np.float32))
    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.repeat(3, 1, 1)
    image_tensor = image_tensor / 255.0
    
    return image_tensor, image_np


def run_inference_on_image(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    score_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on a single image.
    
    Returns:
        boxes: Predicted boxes in format [N, 4] (x1, y1, x2, y2)
        scores: Confidence scores [N]
        labels: Predicted class labels [N]
    """
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


def bbox_iou_x_axis(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two boxes using ONLY x-axis overlap.
    This is appropriate for OCT B-scans where horizontal position matters
    but vertical extent can vary.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU based on x-axis overlap only
    """
    x1_min, _, x1_max, _ = box1
    x2_min, _, x2_max, _ = box2
    
    # Calculate x-axis intersection
    inter_xmin = max(x1_min, x2_min)
    inter_xmax = min(x1_max, x2_max)
    
    inter_length = max(0, inter_xmax - inter_xmin)
    
    # Calculate x-axis lengths
    box1_length = x1_max - x1_min
    box2_length = x2_max - x2_min
    union_length = box1_length + box2_length - inter_length
    
    if union_length == 0:
        return 0.0
    
    return inter_length / union_length


def coco_bbox_to_xyxy(bbox: List[float]) -> np.ndarray:
    """
    Convert COCO format bbox [x, y, width, height] to [x1, y1, x2, y2].
    """
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h])


def remap_maybe_to_scr(gt_labels: np.ndarray, maybe_class_id: int = 2, scr_class_id: int = 0) -> np.ndarray:
    """
    Remap 'maybe' class to SCR class since model was only trained on fovea/SCR.
    
    Args:
        gt_labels: Ground truth labels
        maybe_class_id: ID of 'maybe' class (default 2)
        scr_class_id: ID of SCR class (default 0)
    
    Returns:
        Remapped labels
    """
    remapped = gt_labels.copy()
    remapped[remapped == maybe_class_id] = scr_class_id
    return remapped


def compute_ap(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    pred_labels: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    iou_threshold: float = 0.5,
    num_classes: int = 2  # Only fovea and SCR
) -> Dict[str, float]:
    """
    Compute Average Precision for a given IoU threshold using x-axis IoU.
    
    Args:
        pred_boxes: List of predicted boxes for each image [N_i, 4]
        pred_scores: List of predicted scores for each image [N_i]
        pred_labels: List of predicted labels for each image [N_i]
        gt_boxes: List of ground truth boxes for each image [M_i, 4]
        gt_labels: List of ground truth labels for each image [M_i]
        iou_threshold: IoU threshold for matching (applied to x-axis IoU)
        num_classes: Number of classes (2 for fovea and SCR)
    
    Returns:
        Dictionary with AP per class and mAP
    """
    # Collect all predictions with image indices
    all_preds = []
    for img_idx, (boxes, scores, labels) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
        for box, score, label in zip(boxes, scores, labels):
            all_preds.append({
                'image_idx': img_idx,
                'box': box,
                'score': score,
                'label': label
            })
    
    all_preds = sorted(all_preds, key=lambda x: x['score'], reverse=True)
    
    ap_per_class = {}
    
    for class_id in range(num_classes):
        class_preds = [p for p in all_preds if p['label'] == class_id]
        
        class_gt_count = sum(
            np.sum(gt_labels[i] == class_id) for i in range(len(gt_labels))
        )
        
        if class_gt_count == 0:
            continue
        
        gt_matched = [
            np.zeros(len(gt_labels[i]), dtype=bool) for i in range(len(gt_labels))
        ]
        
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        
        for pred_idx, pred in enumerate(class_preds):
            img_idx = pred['image_idx']
            pred_box = pred['box']
            
            gt_mask = gt_labels[img_idx] == class_id
            gt_boxes_class = gt_boxes[img_idx][gt_mask]
            gt_matched_class = gt_matched[img_idx][gt_mask]
            
            if len(gt_boxes_class) == 0:
                fp[pred_idx] = 1
                continue
            
            # Use x-axis IoU
            ious = np.array([bbox_iou_x_axis(pred_box, gt_box) for gt_box in gt_boxes_class])
            
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            
            if max_iou >= iou_threshold and not gt_matched_class[max_iou_idx]:
                tp[pred_idx] = 1
                full_gt_idx = np.where(gt_mask)[0][max_iou_idx]
                gt_matched[img_idx][full_gt_idx] = True
            else:
                fp[pred_idx] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / class_gt_count
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        
        ap_per_class[class_id] = ap
    
    if len(ap_per_class) > 0:
        mAP = np.mean(list(ap_per_class.values()))
    else:
        mAP = 0.0
    
    return {
        'mAP': mAP,
        'per_class': ap_per_class
    }


def compute_metrics(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    pred_labels: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    num_classes: int = 2,
    categories_dict: Dict = None
) -> Dict[str, float]:
    """
    Compute all detection metrics using x-axis IoU.
    
    Returns:
        Dictionary with precision, recall, mAP@0.5, mAP@0.5:0.95, and per-class metrics
    """
    # Compute mAP@0.5 with x-axis IoU
    ap_50 = compute_ap(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, 
                       iou_threshold=0.5, num_classes=num_classes)
    
    # Compute mAP@0.5:0.95 (average over IoU thresholds from 0.5 to 0.95)
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    ap_all = []
    for iou_thresh in iou_thresholds:
        ap = compute_ap(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, 
                       iou_threshold=iou_thresh, num_classes=num_classes)
        ap_all.append(ap['mAP'])
    
    mAP_50_95 = np.mean(ap_all)
    
    # Compute precision and recall at IoU=0.5 using x-axis IoU
    total_tp = 0
    total_fp = 0
    total_gt = sum(len(boxes) for boxes in gt_boxes)
    
    # Per-class metrics
    class_tp = {i: 0 for i in range(num_classes)}
    class_fp = {i: 0 for i in range(num_classes)}
    class_fn = {i: 0 for i in range(num_classes)}
    class_gt_count = {i: 0 for i in range(num_classes)}
    
    for img_idx in range(len(pred_boxes)):
        pred_boxes_img = pred_boxes[img_idx]
        pred_labels_img = pred_labels[img_idx]
        gt_boxes_img = gt_boxes[img_idx]
        gt_labels_img = gt_labels[img_idx]
        
        # Count ground truths per class
        for gt_label in gt_labels_img:
            class_gt_count[gt_label] = class_gt_count.get(gt_label, 0) + 1
        
        gt_matched = np.zeros(len(gt_boxes_img), dtype=bool)
        
        for pred_box, pred_label in zip(pred_boxes_img, pred_labels_img):
            best_iou = 0
            best_idx = -1
            
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes_img, gt_labels_img)):
                if pred_label == gt_label and not gt_matched[gt_idx]:
                    iou = bbox_iou_x_axis(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gt_idx
            
            if best_iou >= 0.5 and best_idx >= 0:
                total_tp += 1
                class_tp[pred_label] = class_tp.get(pred_label, 0) + 1
                gt_matched[best_idx] = True
            else:
                total_fp += 1
                class_fp[pred_label] = class_fp.get(pred_label, 0) + 1
        
        # Count false negatives
        for gt_idx, gt_label in enumerate(gt_labels_img):
            if not gt_matched[gt_idx]:
                class_fn[gt_label] = class_fn.get(gt_label, 0) + 1
    
    total_fn = total_gt - total_tp
    
    precision = total_tp / (total_tp + total_fp + 1e-10)
    recall = total_tp / (total_tp + total_fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Compute per-class precision and recall
    per_class_metrics = {}
    for class_id in range(num_classes):
        tp = class_tp.get(class_id, 0)
        fp = class_fp.get(class_id, 0)
        fn = class_fn.get(class_id, 0)
        gt_count = class_gt_count.get(class_id, 0)
        
        class_precision = tp / (tp + fp + 1e-10)
        class_recall = tp / (tp + fn + 1e-10)
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-10)
        
        class_name = 'unknown'
        if categories_dict and class_id in categories_dict:
            class_name = categories_dict[class_id]['name']
        
        per_class_metrics[class_id] = {
            'name': class_name,
            'precision': class_precision,
            'recall': class_recall,
            'f1': class_f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'gt_count': gt_count
        }
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mAP@0.5': ap_50['mAP'],
        'mAP@0.5:0.95': mAP_50_95,
        'AP_per_class@0.5': ap_50['per_class'],
        'per_class_metrics': per_class_metrics,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate RCNN model on annotated test dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--annotations', type=str, required=True, help='Path to COCO format annotations JSON')
    parser.add_argument('--images-dir', type=str, required=True, help='Path to images directory')
    parser.add_argument('--score-threshold', type=float, default=0.5, help='Score threshold for predictions')
    parser.add_argument('--output', type=str, default='evaluation_metrics.json', help='Path to save results JSON')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Model was trained with 3 classes (background + fovea + SCR) but we remap GT labels during evaluation
    model_num_classes = 3
    eval_num_classes = 2  # Only fovea and SCR for evaluation metrics
    
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(Path(args.checkpoint), num_classes=model_num_classes, device=device)
    model.eval()
    
    print(f'Loading annotations from {args.annotations}...')
    images_dict, annotations_dict, categories_dict = load_coco_annotations(Path(args.annotations))
    
    print(f'Found {len(images_dict)} images')
    print(f'Found {sum(len(anns) for anns in annotations_dict.values())} annotations')
    print(f'Categories: {categories_dict}')
    print('\nIMPORTANT: Remapping "maybe" (class 2) to SCR (class 0) in ground truth')
    print('Using x-axis IoU for bounding box matching (horizontal overlap only)\n')
    
    pred_boxes_all = []
    pred_scores_all = []
    pred_labels_all = []
    gt_boxes_all = []
    gt_labels_all = []
    
    images_dir = Path(args.images_dir)
    
    for image_id, image_info in tqdm(images_dict.items(), desc='Running inference'):
        original_path = image_info['file_name']
        image_name = extract_image_name(original_path)
        
        image_path = None
        for img_file in images_dir.iterdir():
            if img_file.name.endswith(image_name):
                image_path = img_file
                break
        
        if image_path is None:
            continue
        
        image_tensor, image_np = load_image(image_path)
        
        pred_boxes, pred_scores, pred_labels = run_inference_on_image(
            model, image_tensor, device, args.score_threshold
        )
        
        gt_annotations = annotations_dict.get(image_id, [])
        gt_boxes = []
        gt_labels = []
        
        for ann in gt_annotations:
            bbox_xyxy = coco_bbox_to_xyxy(ann['bbox'])
            gt_boxes.append(bbox_xyxy)
            gt_labels.append(ann['category_id'])
        
        gt_boxes = np.array(gt_boxes) if len(gt_boxes) > 0 else np.zeros((0, 4))
        gt_labels = np.array(gt_labels) if len(gt_labels) > 0 else np.zeros((0,), dtype=int)
        
        # Remap "maybe" to SCR
        gt_labels = remap_maybe_to_scr(gt_labels, maybe_class_id=2, scr_class_id=0)
        
        pred_boxes_all.append(pred_boxes)
        pred_scores_all.append(pred_scores)
        pred_labels_all.append(pred_labels)
        gt_boxes_all.append(gt_boxes)
        gt_labels_all.append(gt_labels)
    
    print('\nComputing metrics with x-axis IoU...')
    
    # Create simplified categories dict first (needed for diagnostics)
    simplified_categories = {
        0: {'id': 0, 'name': 'SCR'},
        1: {'id': 1, 'name': 'fovea'}
    }
    
    # Load model label mapping to understand prediction classes
    checkpoint = torch.load(Path(args.checkpoint), map_location='cpu')
    model_label_mapping = checkpoint.get('label_mapping', {})
    print(f'\nModel label mapping from training: {model_label_mapping}')
    
    # Create reverse mapping: model_pred_class -> eval_class
    # Training had: {0: fovea, 1: SCR} -> Model outputs: {1: fovea, 2: SCR}
    # We want eval classes: {0: SCR, 1: fovea}
    # So we need: model_class 1 -> eval_class 1 (fovea)
    #             model_class 2 -> eval_class 0 (SCR)
    pred_class_remap = {}
    if model_label_mapping:
        # model_label_mapping is {training_label: model_output_class}
        # We need {model_output_class: eval_class}
        for training_label, model_class in model_label_mapping.items():
            training_label = int(training_label)
            model_class = int(model_class)
            # training_label 0 = fovea -> eval_class 1
            # training_label 1 = SCR -> eval_class 0
            if training_label == 0:  # fovea in training
                pred_class_remap[model_class] = 1  # fovea in eval
            elif training_label == 1:  # SCR in training
                pred_class_remap[model_class] = 0  # SCR in eval
    
    print(f'Prediction class remapping: {pred_class_remap}')
    print('(model prediction class -> evaluation class)')
    
    # Remap all prediction labels
    print('\nRemapping prediction labels...')
    for i in range(len(pred_labels_all)):
        if len(pred_labels_all[i]) > 0:
            remapped_labels = np.array([pred_class_remap.get(int(label), int(label)) 
                                       for label in pred_labels_all[i]])
            pred_labels_all[i] = remapped_labels
    
    # Print diagnostic info
    all_pred_labels = np.concatenate([labels for labels in pred_labels_all if len(labels) > 0])
    all_gt_labels = np.concatenate([labels for labels in gt_labels_all if len(labels) > 0])
    print(f'\nPrediction label distribution (after remapping):')
    for class_id in range(eval_num_classes):
        count = np.sum(all_pred_labels == class_id)
        class_name = simplified_categories[class_id]['name']
        print(f'  Class {class_id} ({class_name}): {count} predictions')
    print(f'\nGround truth label distribution (after maybe->SCR remapping):')
    for class_id in range(eval_num_classes):
        count = np.sum(all_gt_labels == class_id)
        class_name = simplified_categories[class_id]['name']
        print(f'  Class {class_id} ({class_name}): {count} instances')
    
    metrics = compute_metrics(
        pred_boxes_all, pred_scores_all, pred_labels_all,
        gt_boxes_all, gt_labels_all,
        num_classes=eval_num_classes,
        categories_dict=simplified_categories
    )
    
    # Print results
    print('\n' + '='*60)
    print('EVALUATION RESULTS (with x-axis IoU)')
    print('='*60)
    print(f'Overall Precision:  {metrics["precision"]:.4f}')
    print(f'Overall Recall:     {metrics["recall"]:.4f}')
    print(f'Overall F1:         {metrics["f1"]:.4f}')
    print(f'mAP@0.5:            {metrics["mAP@0.5"]:.4f}')
    print(f'mAP@0.5:0.95:       {metrics["mAP@0.5:0.95"]:.4f}')
    print(f'\nTotal TP: {metrics["total_tp"]}, FP: {metrics["total_fp"]}, FN: {metrics["total_fn"]}')
    
    print('\n' + '-'*60)
    print('Per-Class Metrics:')
    print('-'*60)
    for class_id, class_metrics in metrics['per_class_metrics'].items():
        print(f"\nClass {class_id} ({class_metrics['name']}):")
        print(f"  Precision: {class_metrics['precision']:.4f}")
        print(f"  Recall:    {class_metrics['recall']:.4f}")
        print(f"  F1:        {class_metrics['f1']:.4f}")
        print(f"  AP@0.5:    {metrics['AP_per_class@0.5'].get(class_id, 0.0):.4f}")
        print(f"  TP: {class_metrics['tp']}, FP: {class_metrics['fp']}, FN: {class_metrics['fn']}, GT: {class_metrics['gt_count']}")
    
    print('='*60)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'evaluation_config': {
            'checkpoint': str(args.checkpoint),
            'annotations': str(args.annotations),
            'images_dir': str(args.images_dir),
            'score_threshold': args.score_threshold,
            'model_num_classes': model_num_classes,
            'eval_num_classes': eval_num_classes,
            'iou_type': 'x_axis_only',
            'label_remapping': 'maybe (class 2) -> SCR (class 0)'
        },
        'overall_metrics': {
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'mAP@0.5': float(metrics['mAP@0.5']),
            'mAP@0.5:0.95': float(metrics['mAP@0.5:0.95']),
            'total_tp': int(metrics['total_tp']),
            'total_fp': int(metrics['total_fp']),
            'total_fn': int(metrics['total_fn'])
        },
        'per_class_metrics': {
            int(k): {
                'name': v['name'],
                'precision': float(v['precision']),
                'recall': float(v['recall']),
                'f1': float(v['f1']),
                'AP@0.5': float(metrics['AP_per_class@0.5'].get(k, 0.0)),
                'tp': int(v['tp']),
                'fp': int(v['fp']),
                'fn': int(v['fn']),
                'gt_count': int(v['gt_count'])
            } for k, v in metrics['per_class_metrics'].items()
        },
        'dataset_info': {
            'num_images': len(images_dict),
            'total_ground_truths': sum(len(boxes) for boxes in gt_boxes_all),
            'total_predictions': sum(len(boxes) for boxes in pred_boxes_all)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nResults saved to {output_path}')


if __name__ == '__main__':
    main()
