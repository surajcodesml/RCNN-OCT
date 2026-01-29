"""
Visualize model predictions on Nemours dataset with NMS for overlapping predictions.

This script:
1. Loads inference results from pickle file
2. Applies Non-Maximum Suppression (NMS) to handle overlapping predictions
3. Selects the prediction with highest confidence when multiple predictions overlap
4. Visualizes predictions on B-scan images
5. Saves visualizations organized by volume
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from tqdm import tqdm


def load_hdf5_volumes(hdf5_path: Path) -> Tuple[Dict, tuple]:
    """
    Load HDF5 file and organize B-scans by volume.
    
    Returns:
        volumes: Dictionary mapping volume_id to list of (image, bscan_idx) tuples
        data_shape: Shape of the dataset (num_images, height, width)
    """
    with h5py.File(hdf5_path, 'r') as f:
        images = f['images'][:]
        names = f['names'][:].astype(str)
        bscan_indices = f['bscan_indices'][:]
    
    volumes = {}
    for i, (name, bscan_idx) in enumerate(zip(names, bscan_indices)):
        # Extract volume ID from name (e.g., "256_L_1_1.e2e" -> "256_L_1_1")
        volume_id = name.replace('.e2e', '')
        
        if volume_id not in volumes:
            volumes[volume_id] = []
        
        volumes[volume_id].append({
            'image': images[i],
            'bscan_idx': bscan_idx,
            'global_idx': i
        })
    
    return volumes, images.shape


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two boxes in [x1, y1, x2, y2] format.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    Apply Non-Maximum Suppression to remove overlapping predictions.
    
    For overlapping boxes of the same class, keep only the one with highest confidence.
    
    Args:
        boxes: Array of shape (N, 4) in [x1, y1, x2, y2] format
        scores: Array of shape (N,) with confidence scores
        labels: Array of shape (N,) with class labels
        iou_threshold: IoU threshold for considering boxes as overlapping
    
    Returns:
        keep_indices: Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)
    
    # Sort by score in descending order
    order = scores.argsort()[::-1]
    
    keep = []
    
    while len(order) > 0:
        # Keep the box with highest score
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in order[1:]])
        
        # Keep boxes that don't overlap significantly OR are of different class
        different_class = labels[order[1:]] != labels[i]
        low_iou = ious < iou_threshold
        
        # Keep if either different class OR low IoU
        keep_mask = different_class | low_iou
        
        order = order[1:][keep_mask]
    
    return np.array(keep, dtype=int)


def apply_nms_to_predictions(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply NMS to predictions and return filtered results.
    """
    if len(pred_boxes) == 0:
        return pred_boxes, pred_scores, pred_labels
    
    keep_indices = nms(pred_boxes, pred_scores, pred_labels, iou_threshold)
    
    return pred_boxes[keep_indices], pred_scores[keep_indices], pred_labels[keep_indices]


def visualize_predictions(
    image_np: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    save_path: Path,
    title: str = "",
    label_mapping: Dict[int, str] = None
) -> None:
    """
    Visualize predictions on B-scan image.
    
    Color coding:
        - Label 1 (Fovea): Blue
        - Label 2 (SCR): Red
    """
    if label_mapping is None:
        label_mapping = {1: 'fovea', 2: 'SCR'}
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.imshow(image_np, cmap='gray')
    
    # Colors for each class
    colors = {1: 'blue', 2: 'red'}
    
    # Draw predictions
    for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = pred_box
        width = x2 - x1
        height = y2 - y1
        
        color = colors.get(pred_label, 'cyan')
        class_name = label_mapping.get(pred_label, f'class_{pred_label}')
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none',
            linestyle='-'
        )
        ax.add_patch(rect)
        
        # Add label with score
        ax.text(
            x1, y1 - 5, 
            f'{class_name}: {pred_score:.3f}',
            color=color, fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    ax.set_title(title, fontsize=14, weight='bold', pad=10)
    ax.axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=3, label='Fovea'),
        Line2D([0], [0], color='red', linewidth=3, label='SCR')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_volume_predictions(
    volume_id: str,
    volume_data: List[Dict],
    inference_results: List[Dict],
    output_dir: Path,
    nms_iou_threshold: float = 0.5,
    visualize_all: bool = False,
    max_visualizations: int = 10
) -> Dict:
    """
    Process predictions for a single volume.
    
    Args:
        volume_id: Volume identifier
        volume_data: List of B-scan data for this volume
        inference_results: List of inference results for this volume
        output_dir: Output directory for visualizations
        nms_iou_threshold: IoU threshold for NMS
        visualize_all: If True, visualize all B-scans; otherwise sample
        max_visualizations: Maximum number of B-scans to visualize per volume
    
    Returns:
        Dictionary with volume statistics
    """
    volume_output_dir = output_dir / volume_id
    volume_output_dir.mkdir(parents=True, exist_ok=True)
    
    total_predictions_before_nms = 0
    total_predictions_after_nms = 0
    bscans_with_predictions = 0
    
    # Determine which B-scans to visualize
    num_bscans = len(volume_data)
    if visualize_all or num_bscans <= max_visualizations:
        bscan_indices_to_viz = list(range(num_bscans))
    else:
        # Sample evenly across the volume
        step = num_bscans / max_visualizations
        bscan_indices_to_viz = [int(i * step) for i in range(max_visualizations)]
    
    for idx in bscan_indices_to_viz:
        bscan_data = volume_data[idx]
        result = inference_results[idx]
        
        image_np = bscan_data['image']
        bscan_idx = bscan_data['bscan_idx']
        
        pred_boxes = result['pred_boxes']
        pred_scores = result['pred_scores']
        pred_labels = result['pred_labels']
        
        num_before = len(pred_boxes)
        total_predictions_before_nms += num_before
        
        # Apply NMS
        pred_boxes_nms, pred_scores_nms, pred_labels_nms = apply_nms_to_predictions(
            pred_boxes, pred_scores, pred_labels, nms_iou_threshold
        )
        
        num_after = len(pred_boxes_nms)
        total_predictions_after_nms += num_after
        
        if num_after > 0:
            bscans_with_predictions += 1
        
        # Visualize
        title = f'{volume_id} - B-scan {bscan_idx:03d}\n'
        title += f'Predictions: {num_after} (before NMS: {num_before})'
        
        save_path = volume_output_dir / f'bscan_{bscan_idx:03d}.png'
        
        visualize_predictions(
            image_np, pred_boxes_nms, pred_labels_nms, pred_scores_nms,
            save_path, title
        )
    
    return {
        'volume_id': volume_id,
        'num_bscans': num_bscans,
        'num_visualized': len(bscan_indices_to_viz),
        'total_predictions_before_nms': total_predictions_before_nms,
        'total_predictions_after_nms': total_predictions_after_nms,
        'bscans_with_predictions': bscans_with_predictions
    }


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Nemours dataset predictions with NMS'
    )
    parser.add_argument(
        '--inference-pkl', type=str, required=True,
        help='Path to inference results pickle file'
    )
    parser.add_argument(
        '--hdf5-path', type=str, required=True,
        help='Path to HDF5 dataset file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='nemours_visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--nms-iou-threshold', type=float, default=0.5,
        help='IoU threshold for NMS (default: 0.5)'
    )
    parser.add_argument(
        '--max-volumes', type=int, default=None,
        help='Maximum number of volumes to process (default: all)'
    )
    parser.add_argument(
        '--max-viz-per-volume', type=int, default=10,
        help='Maximum visualizations per volume (default: 10)'
    )
    parser.add_argument(
        '--visualize-all', action='store_true',
        help='Visualize all B-scans in each volume'
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('='*60)
    print('Nemours Dataset Prediction Visualization with NMS')
    print('='*60)
    print(f'Inference results: {args.inference_pkl}')
    print(f'HDF5 dataset: {args.hdf5_path}')
    print(f'Output directory: {output_dir}')
    print(f'NMS IoU threshold: {args.nms_iou_threshold}')
    print(f'Max visualizations per volume: {args.max_viz_per_volume}')
    print('='*60)
    
    # Load HDF5 dataset
    print('\nLoading HDF5 dataset...')
    volumes, data_shape = load_hdf5_volumes(Path(args.hdf5_path))
    print(f'Loaded {len(volumes)} volumes with {data_shape[0]} total B-scans')
    print(f'Image dimensions: {data_shape[1]} x {data_shape[2]}')
    
    # Load inference results
    print(f'\nLoading inference results from {args.inference_pkl}...')
    with open(args.inference_pkl, 'rb') as f:
        # Load in chunks to avoid memory issues
        inference_data = pickle.load(f)
    
    print(f'Loaded inference results for {len(inference_data)} volumes')
    
    # Process volumes
    volume_ids = list(volumes.keys())
    if args.max_volumes:
        volume_ids = volume_ids[:args.max_volumes]
    
    print(f'\nProcessing {len(volume_ids)} volumes...\n')
    
    all_stats = []
    
    for volume_id in tqdm(volume_ids, desc='Processing volumes'):
        if volume_id not in inference_data:
            print(f'Warning: No inference results for volume {volume_id}')
            continue
        
        volume_data = volumes[volume_id]
        inference_results = inference_data[volume_id]
        
        stats = process_volume_predictions(
            volume_id, volume_data, inference_results,
            output_dir, args.nms_iou_threshold,
            args.visualize_all, args.max_viz_per_volume
        )
        
        all_stats.append(stats)
    
    # Print summary statistics
    print('\n' + '='*60)
    print('SUMMARY STATISTICS')
    print('='*60)
    
    total_bscans = sum(s['num_bscans'] for s in all_stats)
    total_visualized = sum(s['num_visualized'] for s in all_stats)
    total_pred_before = sum(s['total_predictions_before_nms'] for s in all_stats)
    total_pred_after = sum(s['total_predictions_after_nms'] for s in all_stats)
    total_bscans_with_pred = sum(s['bscans_with_predictions'] for s in all_stats)
    
    print(f'Total volumes processed: {len(all_stats)}')
    print(f'Total B-scans: {total_bscans}')
    print(f'Total B-scans visualized: {total_visualized}')
    print(f'B-scans with predictions: {total_bscans_with_pred}')
    print(f'\nPredictions before NMS: {total_pred_before}')
    print(f'Predictions after NMS: {total_pred_after}')
    print(f'Predictions removed by NMS: {total_pred_before - total_pred_after}')
    print(f'Reduction: {100 * (total_pred_before - total_pred_after) / max(total_pred_before, 1):.1f}%')
    
    # Print per-volume statistics
    print('\n' + '-'*60)
    print('Per-Volume Statistics:')
    print('-'*60)
    print(f'{"Volume ID":<25} {"B-scans":<10} {"Viz":<8} {"Pred (Before)":<15} {"Pred (After)":<15}')
    print('-'*60)
    
    for stats in all_stats[:20]:  # Show first 20 volumes
        print(
            f'{stats["volume_id"]:<25} '
            f'{stats["num_bscans"]:<10} '
            f'{stats["num_visualized"]:<8} '
            f'{stats["total_predictions_before_nms"]:<15} '
            f'{stats["total_predictions_after_nms"]:<15}'
        )
    
    if len(all_stats) > 20:
        print(f'... and {len(all_stats) - 20} more volumes')
    
    print('='*60)
    print(f'\n✓ Visualizations saved to: {output_dir}/')
    print('='*60)


if __name__ == '__main__':
    main()
