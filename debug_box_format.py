"""Check bounding box format and coordinates."""
import pickle
from pathlib import Path
import numpy as np

def analyze_box_format(root_path):
    """Analyze the format of bounding boxes."""
    root = Path(root_path)
    all_files = sorted(root.rglob("*.pkl"))
    
    print("Analyzing bounding box format...\n")
    
    samples_with_label_1 = []
    
    # Find files with label=1 (actual targets)
    for pkl_path in all_files[:2000]:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        box = data['box']
        label = data['label']
        
        if hasattr(box, 'numpy'):
            box = box.numpy()
        if hasattr(label, 'numpy'):
            label = label.numpy()
        
        box = np.asarray(box)
        label = np.asarray(label)
        
        # Find samples with label=1 and non-zero boxes
        if np.any(label == 1) and not np.all(box == 0):
            samples_with_label_1.append((pkl_path, box, label))
            if len(samples_with_label_1) >= 20:
                break
    
    print(f"Found {len(samples_with_label_1)} samples with label=1\n")
    print("="*80)
    
    for i, (path, box, label) in enumerate(samples_with_label_1[:10]):
        print(f"\nSample {i}: {path.name}")
        print(f"  Boxes shape: {box.shape}")
        print(f"  Labels: {label}")
        print(f"  Box values:")
        for j, b in enumerate(box):
            print(f"    Box {j}: {b} (label={label[j] if j < len(label) else 'N/A'})")
            if len(b) == 4:
                print(f"      [x1={b[0]:.4f}, y1={b[1]:.4f}, x2={b[2]:.4f}, y2={b[3]:.4f}]")
                print(f"      Width: {abs(b[2] - b[0]):.4f}, Height: {abs(b[3] - b[1]):.4f}")
                
                # Check if this looks like xyxy format
                if b[2] > 1.0 or b[3] > 1.0:
                    print(f"      ⚠️  Coordinates > 1.0 detected!")
                if b[2] < b[0] or b[3] < b[1]:
                    print(f"      ⚠️  x2 < x1 or y2 < y1 - might be xywh format or need conversion")
    
    print("\n" + "="*80)
    print("Analysis:")
    print("="*80)
    
    # Analyze format
    all_boxes = []
    for _, box, label in samples_with_label_1:
        for j, b in enumerate(box):
            if j < len(label) and label[j] == 1:
                all_boxes.append(b)
    
    if all_boxes:
        all_boxes = np.array(all_boxes)
        print(f"\nCoordinate ranges for label=1 boxes:")
        print(f"  Col 0 (x1/cx): [{all_boxes[:, 0].min():.4f}, {all_boxes[:, 0].max():.4f}]")
        print(f"  Col 1 (y1/cy): [{all_boxes[:, 1].min():.4f}, {all_boxes[:, 1].max():.4f}]")
        print(f"  Col 2 (x2/w):  [{all_boxes[:, 2].min():.4f}, {all_boxes[:, 2].max():.4f}]")
        print(f"  Col 3 (y2/h):  [{all_boxes[:, 3].min():.4f}, {all_boxes[:, 3].max():.4f}]")
        
        # Check if columns 2 and 3 look like widths/heights (small values)
        if all_boxes[:, 2].max() < 0.5 and all_boxes[:, 3].max() < 0.5:
            print("\n⚠️  Columns 2 & 3 have small values (<0.5)")
            print("   This suggests format is [x_center, y_center, width, height]")
            print("   Need to convert to [x1, y1, x2, y2] format")
        
        # Check if any coords > 1.0
        if np.any(all_boxes > 1.0):
            print("\n⚠️  Some coordinates > 1.0 detected!")
            print("   This suggests coordinates might be in pixel space, not normalized [0,1]")
            print("   Or there's an issue with the data format")

if __name__ == "__main__":
    data_root = "/home/suraj/Data/Nemours/pickle"
    analyze_box_format(data_root)
