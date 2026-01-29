"""
Visualize model predictions on Nemours dataset with live inference and NMS.

This memory-efficient version:
1. Loads volumes from HDF5 one at a time
2. Runs inference on-the-fly
3. Applies NMS to handle overlapping predictions
4. Generates visualizations without loading entire dataset into memory
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from tqdm import tqdm

from inference import load_model


def load_hdf5_volume_info(hdf5_path: Path) -> Tuple[Dict, tuple]:
    """
    Load HDF5 file metadata and organize indices by volume.
    
    Returns:
        volume_indices: Dictionary mapping volume_id to list of global indices
        data_shape: Shape of the dataset (num_images, height, width)
    """
    with h5py.File(hdf5_path, 'r') as f:
        names = f['names'][:].astype(str)
        bscan_indices = f['bscan_indices'][:]
        data_shape = f['images'].shape
    
    volume_indices = {}
    for i, (name, bscan_idx) in enumerate(zip(names, bscan_indices)):
        # Extract volume ID from name (e.g., "256_L_1_1.e2e" -> "256_L_1_1")
        volume_id = name.replace('.e2e', '')
        
        if volume_id not in volume_indices:
            volume_indices[volume_id] = []
        
        volume_indices[volume_id].append({
            'global_idx': i,
            'bscan_idx': bscan_idx
        })
    
    return volume_indices, data_shape


def load_volume_images(hdf5_path: Path, indices: List[int]) -> np.ndarray:
    """
    Load specific images from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        indices: List of global indices to load
    
    Returns:
        images: Array of shape (N, H, W)
    """
    with h5py.File(hdf5_path, 'r') as f:
        images = f['images'][indices]
    return images


def run_inference_on_image(
    model: torch.nn.Module,
    image_np: np.ndarray,
    device: torch.device,
    score_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on a single OCT B-scan image.
    
    Returns:
        pred_boxes: Array of shape (N, 4) in [x1, y1, x2, y2] format
        pred_scores: Array of shape (N,)
        pred_labels: Array of shape (N,)
    """
    # Normalize image
    image_tensor = torch.from_numpy(image_np.astype(np.float32))
    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Convert to 3-channel
    image_tensor = image_tensor.repeat(3, 1, 1)
    image_tensor = image_tensor / 255.0
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model([image_tensor.to(device)])
    
    # Extract predictions
    output = outputs[0]
    pred_boxes = output['boxes'].cpu().numpy()
    pred_scores = output['scores'].cpu().numpy()
    pred_labels = output['labels'].cpu().numpy()
    
    # Filter by score threshold
    keep = pred_scores >= score_threshold
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]
    
    return pred_boxes, pred_scores, pred_labels


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
    
    For overlapping boxes, keep only the one with highest confidence.
    
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


def process_volume(
    volume_id: str,
    volume_info: List[Dict],
    hdf5_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    score_threshold: float = 0.5,
    nms_iou_threshold: float = 0.5,
    visualize_all: bool = False,
    max_visualizations: int = 10
) -> Dict:
    """
    Process predictions for a single volume.
    
    Returns:
        Dictionary with volume statistics
    """
    volume_output_dir = output_dir / volume_id
    volume_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load images for this volume
    global_indices = [info['global_idx'] for info in volume_info]
    images = load_volume_images(hdf5_path, global_indices)
    
    total_predictions_before_nms = 0
    total_predictions_after_nms = 0
    bscans_with_predictions = 0
    
    # Determine which B-scans to visualize
    num_bscans = len(volume_info)
    if visualize_all or num_bscans <= max_visualizations:
        bscan_indices_to_viz = list(range(num_bscans))
    else:
        # Sample evenly across the volume
        step = num_bscans / max_visualizations
        bscan_indices_to_viz = [int(i * step) for i in range(max_visualizations)]
    
    for idx in bscan_indices_to_viz:
        info = volume_info[idx]
        image_np = images[idx]
        bscan_idx = info['bscan_idx']
        
        # Run inference
        pred_boxes, pred_scores, pred_labels = run_inference_on_image(
            model, image_np, device, score_threshold
        )
        
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
        description='Visualize Nemours dataset predictions with live inference and NMS'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--hdf5-path', type=str, required=True,
        help='Path to HDF5 dataset file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='nemours_visualizations_live',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--score-threshold', type=float, default=0.5,
        help='Score threshold for predictions (default: 0.5)'
    )
    parser.add_argument(
        '--nms-iou-threshold', type=float, default=0.5,
        help='IoU threshold for NMS (default: 0.5)'
    )
    parser.add_argument(
        '--num-classes', type=int, default=3,
        help='Number of classes in model (default: 3)'
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('='*60)
    print('Nemours Dataset Prediction Visualization (Live Inference)')
    print('='*60)
    print(f'Model checkpoint: {args.checkpoint}')
    print(f'HDF5 dataset: {args.hdf5_path}')
    print(f'Output directory: {output_dir}')
    print(f'Device: {device}')
    print(f'Score threshold: {args.score_threshold}')
    print(f'NMS IoU threshold: {args.nms_iou_threshold}')
    print(f'Max visualizations per volume: {args.max_viz_per_volume}')
    print('='*60)
    
    # Load model
    print('\nLoading model...')
    model = load_model(Path(args.checkpoint), num_classes=args.num_classes, device=device)
    model.eval()
    print('✓ Model loaded successfully')
    
    # Load HDF5 metadata
    print('\nLoading HDF5 metadata...')
    volume_indices, data_shape = load_hdf5_volume_info(Path(args.hdf5_path))
    print(f'✓ Found {len(volume_indices)} volumes with {data_shape[0]} total B-scans')
    print(f'  Image dimensions: {data_shape[1]} x {data_shape[2]}')
    
    # Process volumes
    volume_ids = list(volume_indices.keys())
    if args.max_volumes:
        volume_ids = volume_ids[:args.max_volumes]
    
    print(f'\nProcessing {len(volume_ids)} volumes...\n')
    
    all_stats = []
    
    for volume_id in tqdm(volume_ids, desc='Processing volumes'):
        volume_info = volume_indices[volume_id]
        
        stats = process_volume(
            volume_id, volume_info, Path(args.hdf5_path),
            model, device, output_dir,
            args.score_threshold, args.nms_iou_threshold,
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
    if total_pred_before > 0:
        print(f'Reduction: {100 * (total_pred_before - total_pred_after) / total_pred_before:.1f}%')
    
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
