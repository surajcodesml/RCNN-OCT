"""Check the actual content of box and label tensors."""
import pickle
from pathlib import Path
import numpy as np
import torch

def inspect_tensor_content(root_path):
    """Inspect the actual content of box and label tensors."""
    root = Path(root_path)
    all_files = sorted(root.rglob("*.pkl"))
    
    print("Inspecting tensor content from pickle files...\n")
    
    files_with_annotations = 0
    files_without_annotations = 0
    label_counts = {}
    
    # Check center files that should have annotations
    target_patterns = ['012.pkl', '013.pkl', '014.pkl', '015.pkl', '016.pkl']
    
    for pattern in target_patterns:
        matching_files = [f for f in all_files if f.name.endswith(pattern)]
        
        print(f"\n{'='*80}")
        print(f"Checking files ending with {pattern} ({len(matching_files)} files)")
        print('='*80)
        
        for i, pkl_path in enumerate(matching_files[:20]):  # Check first 20
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            img = data['img']
            box = data['box']
            label = data['label']
            name = data['name']
            
            # Convert to numpy if torch tensors
            if isinstance(box, torch.Tensor):
                box = box.numpy()
            if isinstance(label, torch.Tensor):
                label = label.numpy()
            
            box = np.asarray(box)
            label = np.asarray(label)
            
            # Count labels
            if label.size > 0 and not (label.size == 1 and label.item() == 0):
                for lbl in label.flatten():
                    if lbl != 0:  # Ignore background/padding
                        label_counts[int(lbl)] = label_counts.get(int(lbl), 0) + 1
            
            has_annotation = box.size > 0 and label.size > 0
            
            if has_annotation and np.any(label != 0):
                files_with_annotations += 1
                print(f"\n✓ {pkl_path.name}")
                print(f"  Image: {type(img)}, shape: {tuple(img.shape) if hasattr(img, 'shape') else 'N/A'}")
                print(f"  Box shape: {box.shape}, dtype: {box.dtype}")
                print(f"  Box content: {box}")
                print(f"  Label shape: {label.shape}, dtype: {label.dtype}")
                print(f"  Label content: {label}")
                print(f"  Name: {name}")
            else:
                files_without_annotations += 1
                if i < 3:  # Show first 3 without annotations
                    print(f"\n✗ {pkl_path.name}")
                    print(f"  Box: {box} (shape: {box.shape})")
                    print(f"  Label: {label} (shape: {label.shape})")
        
        if files_with_annotations > 0:
            break  # Found some, no need to check all patterns
    
    # Also check some random files
    print(f"\n{'='*80}")
    print("Checking random sample of ALL files...")
    print('='*80)
    
    for i in [100, 500, 1000, 2000, 5000]:
        if i < len(all_files):
            pkl_path = all_files[i]
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            box = data['box']
            label = data['label']
            
            if isinstance(box, torch.Tensor):
                box = box.numpy()
            if isinstance(label, torch.Tensor):
                label = label.numpy()
            
            box = np.asarray(box)
            label = np.asarray(label)
            
            if np.any(label != 0):
                print(f"\n✓ Index {i}: {pkl_path.name}")
                print(f"  Box: {box}")
                print(f"  Label: {label}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"Files with annotations: {files_with_annotations}")
    print(f"Files without annotations: {files_without_annotations}")
    print(f"Label distribution: {label_counts}")
    
    if files_with_annotations == 0:
        print("\n⚠️  Issue: The 'box' and 'label' keys exist but appear to be empty/zero tensors")
        print("The pickle files have the structure but no actual annotation data")

if __name__ == "__main__":
    data_root = "/home/suraj/Data/Nemours/pickle"
    inspect_tensor_content(data_root)
