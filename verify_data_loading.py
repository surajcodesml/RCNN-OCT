"""Verification script to test correct data loading and bbox conversion."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from dataset import create_datasets


def visualize_sample(image_tensor: torch.Tensor, target: dict, title: str = "Sample") -> None:
    """Visualize a single sample with bounding boxes.
    
    Args:
        image_tensor: Image tensor in CHW format (3, H, W)
        target: Target dictionary with boxes and labels
        title: Plot title
    """
    # Convert to numpy for visualization (use first channel)
    img_array = image_tensor[0].numpy()
    boxes = target["boxes"].numpy()
    labels = target["labels"].numpy()
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_array, cmap='gray')
    
    # Define colors for different labels
    colors = {1: 'red', 2: 'blue'}
    label_names = {1: 'Class 1', 2: 'Class 2'}
    
    # Plot bounding boxes (already in pixel coordinates and corner format)
    for box, label in zip(boxes, labels):
        label_val = int(label)
        x1, y1, x2, y2 = box
        
        box_width = x2 - x1
        box_height = y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), box_width, box_height,
            linewidth=2, edgecolor=colors.get(label_val, 'yellow'),
            facecolor='none',
            label=label_names.get(label_val, f'Class {label_val}')
        )
        ax.add_patch(rect)
        
        # Add label text at center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        ax.text(center_x, center_y, str(label_val), 
                color=colors.get(label_val, 'yellow'), 
                fontsize=12, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    if len(boxes) > 0:
        ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def verify_data_loading(data_root: Path, num_samples: int = 5) -> None:
    """Verify that data loading and bbox conversion is working correctly.
    
    Args:
        data_root: Root directory containing pickle files
        num_samples: Number of samples to verify
    """
    print(f"Loading datasets from {data_root}...")
    train_dataset, val_dataset, label_mapping = create_datasets(
        data_root,
        val_ratio=0.2,
        seed=42,
        max_samples=100  # Limit for quick testing
    )
    
    print(f"\nDataset Info:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Number of classes: {train_dataset.num_classes}")
    print(f"  Label mapping: {label_mapping}")
    
    print(f"\n{'='*60}")
    print("Verifying bounding box conversion from YOLO to corner format")
    print('='*60)
    
    # Test samples with boxes
    samples_checked = 0
    for idx in range(len(train_dataset)):
        if samples_checked >= num_samples:
            break
            
        image, target = train_dataset[idx]
        
        # Skip samples without boxes
        if target["boxes"].shape[0] == 0:
            continue
        
        samples_checked += 1
        
        print(f"\nSample {idx}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Number of boxes: {target['boxes'].shape[0]}")
        print(f"  Labels: {target['labels'].tolist()}")
        print(f"  Box format: [x1, y1, x2, y2] (pixel coordinates)")
        
        for i, (box, label) in enumerate(zip(target["boxes"], target["labels"])):
            x1, y1, x2, y2 = box.tolist()
            print(f"    Box {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], Label: {label.item()}")
            
            # Verify box is within image bounds
            _, h, w = image.shape
            assert 0 <= x1 <= w, f"x1 out of bounds: {x1}"
            assert 0 <= x2 <= w, f"x2 out of bounds: {x2}"
            assert 0 <= y1 <= h, f"y1 out of bounds: {y1}"
            assert 0 <= y2 <= h, f"y2 out of bounds: {y2}"
            assert x1 < x2, f"x1 >= x2: {x1} >= {x2}"
            assert y1 < y2, f"y1 >= y2: {y1} >= {y2}"
        
        print(f"  ✓ All boxes valid (within bounds, correct format)")
        
        # Visualize the sample
        visualize_sample(image, target, f"Train Sample {idx}")
    
    print(f"\n{'='*60}")
    print(f"✓ Verification complete! Checked {samples_checked} samples with bounding boxes")
    print(f"  - All boxes are in correct corner format [x1, y1, x2, y2]")
    print(f"  - All coordinates are in pixel space")
    print(f"  - All boxes are within image boundaries")
    print('='*60)


def main():
    # Default data root
    data_root = Path("/home/suraj/Data/Nemours/pickle")
    
    if len(sys.argv) > 1:
        data_root = Path(sys.argv[1])
    
    if not data_root.exists():
        print(f"Error: Data root {data_root} does not exist")
        sys.exit(1)
    
    verify_data_loading(data_root, num_samples=5)


if __name__ == "__main__":
    main()
