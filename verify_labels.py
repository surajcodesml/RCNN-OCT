"""Verify that labels 0 and 1 are being loaded correctly."""
from pathlib import Path
from dataset import create_datasets

def verify_labels():
    """Check that fovea (0) and SCR (1) labels are present."""
    data_root = Path("/home/suraj/Data/Nemours/pickle")
    
    print("Loading datasets...")
    train_ds, val_ds, label_mapping = create_datasets(
        data_root,
        val_ratio=0.2,
        seed=42,
        filter_empty_ratio=0.0,
        max_samples=200
    )
    
    print(f"\nLabel mapping: {label_mapping}")
    print("Expected: {{0: 1, 1: 2}} (0=Fovea, 1=SCR mapped to class IDs 1, 2)")
    
    # Check samples with each label
    label_0_count = 0
    label_1_count = 0
    label_2_count = 0
    empty_count = 0
    
    print(f"\nChecking {len(train_ds)} training samples...")
    for i in range(len(train_ds)):
        img, target = train_ds[i]
        labels = target['labels'].numpy()
        boxes = target['boxes'].numpy()
        
        if len(boxes) == 0:
            empty_count += 1
        else:
            # These are remapped labels (1 or 2), so check original
            # by looking at what we expect
            if len(labels) > 0:
                if 1 in labels:  # Remapped from 0 (Fovea)
                    label_0_count += 1
                if 2 in labels:  # Remapped from 1 (SCR)
                    label_1_count += 1
        
        # Show first few samples with boxes
        if i < 10 and len(boxes) > 0:
            print(f"  Sample {i}: {len(boxes)} boxes, labels={labels}")
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Samples with Fovea (label 0 → class 1): {label_0_count}")
    print(f"Samples with SCR (label 1 → class 2): {label_1_count}")
    print(f"Samples with no boxes: {empty_count}")
    print(f"Total samples: {len(train_ds)}")
    
    if label_0_count == 0:
        print("\n⚠️  WARNING: No Fovea samples found! Label 0 might still be filtered.")
    else:
        print(f"\n✓ Fovea samples present ({label_0_count} samples)")
    
    if label_1_count == 0:
        print("⚠️  WARNING: No SCR samples found!")
    else:
        print(f"✓ SCR samples present ({label_1_count} samples)")
    
    # Check a specific sample with label info
    print(f"\n{'='*60}")
    print("Detailed check of samples with boxes:")
    print(f"{'='*60}")
    count = 0
    for i in range(len(train_ds)):
        img, target = train_ds[i]
        if len(target['boxes']) > 0:
            print(f"Sample {i}: boxes={len(target['boxes'])}, labels={target['labels'].tolist()}")
            count += 1
            if count >= 5:
                break

if __name__ == "__main__":
    verify_labels()
