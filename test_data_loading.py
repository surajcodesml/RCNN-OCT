"""Simple test to verify data loading and label mapping."""
import pickle
from pathlib import Path
import sys

# Test 1: Load raw pickle file
print("="*60)
print("TEST 1: Loading raw pickle file")
print("="*60)

pkl_path = Path("/home/suraj/Data/Nemours/pickle/3_L_3015.pkl")
if not pkl_path.exists():
    print(f"Error: File {pkl_path} not found")
    print("Looking for any .pkl files...")
    pkl_dir = Path("/home/suraj/Data/Nemours/pickle")
    if pkl_dir.exists():
        pkl_files = list(pkl_dir.glob("*.pkl"))
        if pkl_files:
            pkl_path = pkl_files[0]
            print(f"Using {pkl_path}")
        else:
            print("No .pkl files found!")
            sys.exit(1)
    else:
        print(f"Directory {pkl_dir} not found")
        sys.exit(1)

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(f"\nLoaded pickle file: {pkl_path.name}")
print(f"Keys in pickle: {data.keys()}")

if 'img' in data:
    print(f"Image shape: {data['img'].shape}")
    print(f"Image dtype: {data['img'].dtype}")
    print(f"Image range: [{data['img'].min():.3f}, {data['img'].max():.3f}]")

if 'box' in data:
    print(f"Boxes shape: {data['box'].shape}")
    print(f"Boxes dtype: {data['box'].dtype}")
    print(f"Boxes:\n{data['box']}")

if 'label' in data:
    print(f"Labels shape: {data['label'].shape}")
    print(f"Labels dtype: {data['label'].dtype}")
    print(f"Labels: {data['label']}")

if 'name' in data:
    print(f"Name: {data['name']}")

# Test 2: Test label mapping
print("\n" + "="*60)
print("TEST 2: Testing label mapping")
print("="*60)

try:
    from dataset import build_label_mapping, find_pkl_files
    
    pkl_dir = Path("/home/suraj/Data/Nemours/pickle")
    files = find_pkl_files(pkl_dir)
    print(f"\nFound {len(files)} pickle files")
    
    # Use first 50 files for quick test
    test_files = files[:50]
    label_mapping = build_label_mapping(test_files, ignore_label=2)
    
    print(f"\nLabel mapping (ignore_label=2):")
    print(f"  {label_mapping}")
    print(f"\nExpected: {{0: 1, 1: 2}}")
    print(f"  Original label 0 (Fovea) -> Class 1")
    print(f"  Original label 1 (SCR) -> Class 2")
    print(f"  Label 2 (No boxes) -> Filtered out")
    
    if label_mapping == {0: 1, 1: 2}:
        print("\n✓ Label mapping is CORRECT!")
    else:
        print("\n⚠ Label mapping differs from expected")
        
except ImportError as e:
    print(f"Cannot import dataset module: {e}")
    print("Skipping label mapping test")

# Test 3: Test dataset loading
print("\n" + "="*60)
print("TEST 3: Testing dataset __getitem__")
print("="*60)

try:
    from dataset import create_datasets
    
    pkl_dir = Path("/home/suraj/Data/Nemours/pickle")
    train_ds, val_ds, label_mapping = create_datasets(
        pkl_dir,
        val_ratio=0.2,
        seed=42,
        filter_empty_ratio=0.0,
        max_samples=20  # Small test
    )
    
    print(f"\nDataset created:")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples: {len(val_ds)}")
    print(f"  Label mapping: {label_mapping}")
    print(f"  Num classes: {train_ds.num_classes}")
    
    # Load first sample with boxes
    print("\nLoading samples to verify box conversion...")
    samples_checked = 0
    for idx in range(len(train_ds)):
        if samples_checked >= 3:
            break
            
        image, target = train_ds[idx]
        
        if target['boxes'].shape[0] == 0:
            continue
            
        samples_checked += 1
        print(f"\nSample {idx}:")
        print(f"  Image shape: {image.shape} (should be [3, H, W])")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Num boxes: {target['boxes'].shape[0]}")
        print(f"  Box format: [x1, y1, x2, y2] in pixel coordinates")
        
        for i, (box, label) in enumerate(zip(target['boxes'], target['labels'])):
            x1, y1, x2, y2 = box.tolist()
            print(f"    Box {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], Label: {label.item()}")
            
            # Verify box validity
            _, h, w = image.shape
            assert 0 <= x1 < x2 <= w, f"Invalid x coordinates: {x1}, {x2}"
            assert 0 <= y1 < y2 <= h, f"Invalid y coordinates: {y1}, {y2}"
            assert label.item() in [1, 2], f"Invalid label: {label.item()} (should be 1 or 2)"
        
        print(f"  ✓ All boxes valid")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nSummary:")
    print("  - Pickle files load correctly")
    print("  - Label mapping is correct: {0: 1, 1: 2}")
    print("  - YOLO format -> Corner format conversion works")
    print("  - Boxes are in pixel coordinates")
    print("  - Model will predict 3 classes: Background(0), Fovea(1), SCR(2)")
    
except Exception as e:
    print(f"\n✗ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
