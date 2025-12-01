"""Debug script to check specific pickle files that should have annotations."""
import pickle
from pathlib import Path
import numpy as np

def inspect_specific_patterns(root_path):
    """Inspect pickle files with specific endings that should have boxes."""
    root = Path(root_path)
    all_files = sorted(root.rglob("*.pkl"))
    
    print(f"Total pickle files: {len(all_files)}")
    
    # Look for files ending with 012, 013, 014, 015, 016
    target_patterns = ['012.pkl', '013.pkl', '014.pkl', '015.pkl', '016.pkl']
    
    print("\n" + "="*80)
    print("Searching for files ending with 012-016.pkl (center images)...")
    print("="*80)
    
    files_with_boxes = []
    files_without_boxes = []
    total_boxes = 0
    label_distribution = {}
    
    for pattern in target_patterns:
        matching_files = [f for f in all_files if f.name.endswith(pattern)]
        print(f"\nFiles ending with '{pattern}': {len(matching_files)}")
        
        for pkl_path in matching_files[:10]:  # Check first 10 of each pattern
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract data
            if 'image' in data:
                image = data['image']
            elif 'img' in data:
                image = data['img']
            else:
                continue
                
            if 'boxes' in data:
                boxes = data['boxes']
            elif 'bboxes' in data:
                boxes = data['bboxes']
            else:
                boxes = []
                
            if 'labels' in data:
                labels = data['labels']
            else:
                labels = []
            
            boxes = np.asarray(boxes)
            labels = np.asarray(labels)
            
            if len(boxes) > 0:
                files_with_boxes.append(pkl_path.name)
                total_boxes += len(boxes)
                
                # Count label distribution
                for lbl in labels:
                    label_distribution[int(lbl)] = label_distribution.get(int(lbl), 0) + 1
                
                print(f"  ✓ {pkl_path.name}: {len(boxes)} boxes")
                print(f"    Image shape: {np.asarray(image).shape}")
                print(f"    Boxes: {boxes}")
                print(f"    Labels: {labels}")
            else:
                files_without_boxes.append(pkl_path.name)
    
    print("\n" + "="*80)
    print("CHECKING ALL FILES FOR ANY WITH BOXES...")
    print("="*80)
    
    # Sample every 100th file to scan the full dataset quickly
    print("\nScanning every 100th file across entire dataset...")
    for i, pkl_path in enumerate(all_files[::100]):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        boxes = data.get('boxes', data.get('bboxes', []))
        labels = data.get('labels', [])
        
        boxes = np.asarray(boxes)
        labels = np.asarray(labels)
        
        if len(boxes) > 0:
            print(f"  ✓ FOUND BOXES! File: {pkl_path.name}")
            print(f"    Index in dataset: {i*100}")
            print(f"    Boxes: {boxes}")
            print(f"    Labels: {labels}")
            files_with_boxes.append(pkl_path.name)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Files checked with target patterns: {len(files_with_boxes) + len(files_without_boxes)}")
    print(f"Files WITH boxes: {len(files_with_boxes)}")
    print(f"Files WITHOUT boxes: {len(files_without_boxes)}")
    print(f"Total boxes found: {total_boxes}")
    print(f"Label distribution: {label_distribution}")
    
    if len(files_with_boxes) > 0:
        print(f"\nExample files with boxes:")
        for fname in files_with_boxes[:20]:
            print(f"  - {fname}")
    else:
        print("\n⚠️  NO FILES WITH BOUNDING BOXES FOUND!")
        print("\nPossible issues:")
        print("  1. Annotations were never created/saved to pickle files")
        print("  2. Pickle files were created without the annotation step")
        print("  3. Need to check the original data generation script")
        print("  4. Annotations might be in a different file/format")
    
    # Check the actual keys in a few pickle files
    print("\n" + "="*80)
    print("CHECKING PICKLE FILE STRUCTURE (first 5 files)")
    print("="*80)
    for pkl_path in all_files[:5]:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"\n{pkl_path.name}:")
        print(f"  Keys: {list(data.keys())}")
        for key, value in data.items():
            if isinstance(value, (list, np.ndarray)):
                arr = np.asarray(value)
                print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")
            else:
                print(f"    {key}: type={type(value)}")

if __name__ == "__main__":
    data_root = "/home/suraj/Data/Nemours/pickle"
    inspect_specific_patterns(data_root)
