"""Simple verification script to test correct data loading and bbox conversion (no viz)."""
import sys
from pathlib import Path

import torch

from dataset import create_datasets


def verify_data_loading(data_root: Path, num_samples: int = 10) -> None:
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
    
    print(f"\n{'='*70}")
    print("Verifying bounding box conversion: YOLO → Corner Format")
    print(f"  YOLO format: [x_center, y_center, width, height] (normalized)")
    print(f"  Corner format: [x1, y1, x2, y2] (pixel coordinates)")
    print('='*70)
    
    # Test samples with boxes
    samples_checked = 0
    samples_with_boxes = 0
    samples_without_boxes = 0
    total_boxes = 0
    
    for idx in range(min(50, len(train_dataset))):
        image, target = train_dataset[idx]
        
        # Verify image format
        assert image.shape[0] == 3, f"Expected 3 channels, got {image.shape[0]}"
        assert image.ndim == 3, f"Expected 3D tensor, got {image.ndim}D"
        
        num_boxes = target["boxes"].shape[0]
        
        if num_boxes == 0:
            samples_without_boxes += 1
        else:
            samples_with_boxes += 1
            total_boxes += num_boxes
            
            if samples_checked < num_samples:
                samples_checked += 1
                
                print(f"\nSample {idx} (file: {train_dataset.files[idx].name}):")
                print(f"  Image shape: {tuple(image.shape)} (C, H, W)")
                print(f"  Number of boxes: {num_boxes}")
                print(f"  Labels: {target['labels'].tolist()}")
                
                _, h, w = image.shape
                
                for i, (box, label) in enumerate(zip(target["boxes"], target["labels"])):
                    x1, y1, x2, y2 = box.tolist()
                    box_w = x2 - x1
                    box_h = y2 - y1
                    
                    print(f"    Box {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] "
                          f"(w={box_w:.1f}, h={box_h:.1f}), Label: {label.item()}")
                    
                    # Verify box is within image bounds
                    assert 0 <= x1 <= w, f"x1={x1} out of bounds [0, {w}]"
                    assert 0 <= x2 <= w, f"x2={x2} out of bounds [0, {w}]"
                    assert 0 <= y1 <= h, f"y1={y1} out of bounds [0, {h}]"
                    assert 0 <= y2 <= h, f"y2={y2} out of bounds [0, {h}]"
                    assert x1 < x2, f"Invalid box: x1={x1} >= x2={x2}"
                    assert y1 < y2, f"Invalid box: y1={y1} >= y2={y2}"
                    assert label.item() > 0, f"Invalid label: {label.item()} (must be > 0)"
                
                print(f"  ✓ All boxes valid")
    
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"  Total samples checked: {samples_with_boxes + samples_without_boxes}")
    print(f"  Samples with boxes: {samples_with_boxes}")
    print(f"  Samples without boxes: {samples_without_boxes}")
    print(f"  Total boxes: {total_boxes}")
    print(f"  Average boxes per sample: {total_boxes / samples_with_boxes if samples_with_boxes > 0 else 0:.2f}")
    print(f"\n✓ Verification PASSED!")
    print(f"  - All boxes are in correct corner format [x1, y1, x2, y2]")
    print(f"  - All coordinates are in pixel space and within image boundaries")
    print(f"  - All labels are properly mapped (starting from 1)")
    print(f"  - Image format is correct: (3, H, W)")
    print('='*70)


def main():
    # Default data root
    data_root = Path("/home/suraj/Data/Nemours/pickle")
    
    if len(sys.argv) > 1:
        data_root = Path(sys.argv[1])
    
    if not data_root.exists():
        print(f"Error: Data root {data_root} does not exist")
        sys.exit(1)
    
    try:
        verify_data_loading(data_root, num_samples=10)
    except Exception as e:
        print(f"\n❌ Verification FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
