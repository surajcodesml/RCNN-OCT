"""Debug script to inspect pickle files and understand the data structure."""
import pickle
from pathlib import Path
import numpy as np

def inspect_pickle_files(root_path, num_samples=20):
    """Inspect random pickle files to understand data structure."""
    root = Path(root_path)
    pkl_files = sorted(root.rglob("*.pkl"))
    
    print(f"Total pickle files found: {len(pkl_files)}")
    
    # Sample files evenly distributed
    step = max(1, len(pkl_files) // num_samples)
    sample_files = pkl_files[::step][:num_samples]
    
    files_with_boxes = 0
    files_without_boxes = 0
    label_counts = {}
    
    print(f"\nInspecting {len(sample_files)} sample files...\n")
    
    for i, pkl_path in enumerate(sample_files):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract data
        if 'image' in data:
            image = data['image']
        elif 'img' in data:
            image = data['img']
        else:
            print(f"File {i}: {pkl_path.name} - NO IMAGE KEY!")
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
        
        # Count labels
        if len(labels) > 0:
            files_with_boxes += 1
            for lbl in labels:
                label_counts[int(lbl)] = label_counts.get(int(lbl), 0) + 1
        else:
            files_without_boxes += 1
        
        # Print details for first few and any with boxes
        if i < 5 or len(boxes) > 0:
            print(f"File {i}: {pkl_path.name}")
            print(f"  Image shape: {np.asarray(image).shape}")
            print(f"  Boxes shape: {boxes.shape}, count: {len(boxes)}")
            print(f"  Labels shape: {labels.shape}, unique labels: {np.unique(labels) if len(labels) > 0 else 'none'}")
            if len(boxes) > 0:
                print(f"  Box examples: {boxes[:2]}")
                print(f"  Label examples: {labels[:5]}")
            print()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Files with boxes: {files_with_boxes}/{len(sample_files)}")
    print(f"Files without boxes: {files_without_boxes}/{len(sample_files)}")
    print(f"Label distribution across samples: {label_counts}")
    print("="*60)
    
    # Check if all labels are 2 (ignore label)
    if label_counts and all(lbl == 2 for lbl in label_counts.keys()):
        print("\n⚠️  WARNING: All labels are '2' which is being filtered as ignore label!")
        print("This is why no training boxes are present.")
    
    if files_with_boxes == 0:
        print("\n⚠️  CRITICAL: NO FILES FOUND WITH BOUNDING BOXES!")
        print("This explains why loss is 0.0000 - there's nothing to train on.")

if __name__ == "__main__":
    data_root = "/home/suraj/Data/Nemours/pickle"
    inspect_pickle_files(data_root, num_samples=50)
