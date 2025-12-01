"""Quick visual check of data loading - displays samples with boxes."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from dataset import create_datasets
except ImportError:
    print("Error: Cannot import dataset module")
    print("Make sure you're in the RCNN-OCT directory")
    sys.exit(1)

def visualize_sample(image, target, idx, label_mapping):
    """Visualize a single sample with bounding boxes."""
    # Convert to numpy for visualization (use first channel since all 3 are identical)
    img_array = image[0].cpu().numpy() if hasattr(image[0], 'cpu') else image[0].numpy()
    
    boxes = target["boxes"].cpu().numpy() if hasattr(target["boxes"], 'cpu') else target["boxes"].numpy()
    labels = target["labels"].cpu().numpy() if hasattr(target["labels"], 'cpu') else target["labels"].numpy()
    
    # Create reverse mapping for display
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    label_names = {1: 'Fovea', 2: 'SCR'}
    colors = {1: 'red', 2: 'blue'}
    
    fig, ax = plt.subplots(1, figsize=(14, 8))
    ax.imshow(img_array, cmap='gray')
    
    # Plot bounding boxes
    for box, label in zip(boxes, labels):
        label_val = int(label)
        orig_label = reverse_mapping.get(label_val, label_val)
        x1, y1, x2, y2 = box
        
        box_width = x2 - x1
        box_height = y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), box_width, box_height,
            linewidth=3, edgecolor=colors.get(label_val, 'yellow'),
            facecolor='none',
            label=f'{label_names.get(label_val, f"Class {label_val}")} (orig: {orig_label})'
        )
        ax.add_patch(rect)
        
        # Add label text
        ax.text(x1, y1 - 5, label_names.get(label_val, f'C{label_val}'), 
                color=colors.get(label_val, 'yellow'), 
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax.set_title(f'Sample {idx} - {len(boxes)} box(es)', fontsize=16, fontweight='bold')
    ax.axis('off')
    if len(boxes) > 0:
        handles, labels_legend = ax.get_legend_handles_labels()
        # Remove duplicate legend entries
        by_label = dict(zip(labels_legend, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)
    
    plt.tight_layout()
    return fig

def main():
    pkl_dir = Path("/home/suraj/Data/Nemours/pickle")
    
    if not pkl_dir.exists():
        print(f"Error: Directory {pkl_dir} not found")
        sys.exit(1)
    
    print("="*70)
    print("RCNN-OCT Quick Visual Check")
    print("="*70)
    print(f"\nLoading datasets from {pkl_dir}...")
    
    try:
        train_ds, val_ds, label_mapping = create_datasets(
            pkl_dir,
            val_ratio=0.2,
            seed=42,
            filter_empty_ratio=0.0,
            max_samples=100  # Limit for quick check
        )
        
        print(f"\nDataset Info:")
        print(f"  Train samples: {len(train_ds)}")
        print(f"  Val samples: {len(val_ds)}")
        print(f"  Num classes: {train_ds.num_classes}")
        print(f"  Label mapping: {label_mapping}")
        print(f"    → 0 (Fovea) → Class 1")
        print(f"    → 1 (SCR) → Class 2")
        
        # Find samples with boxes
        print(f"\nSearching for samples with bounding boxes...")
        samples_with_boxes = []
        for idx in range(len(train_ds)):
            image, target = train_ds[idx]
            if target['boxes'].shape[0] > 0:
                samples_with_boxes.append((idx, image, target))
                if len(samples_with_boxes) >= 5:
                    break
        
        if not samples_with_boxes:
            print("⚠ No samples with bounding boxes found!")
            return
        
        print(f"Found {len(samples_with_boxes)} samples with boxes")
        print("\nDisplaying samples...")
        
        # Display samples
        for idx, image, target in samples_with_boxes:
            print(f"\nSample {idx}:")
            print(f"  Image shape: {image.shape}")
            print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"  Num boxes: {target['boxes'].shape[0]}")
            
            for i, (box, label) in enumerate(zip(target['boxes'], target['labels'])):
                x1, y1, x2, y2 = box.tolist()
                print(f"    Box {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], "
                      f"Label: {label.item()} ({'Fovea' if label.item() == 1 else 'SCR'})")
            
            fig = visualize_sample(image, target, idx, label_mapping)
            plt.show()
            plt.close(fig)
        
        print("\n" + "="*70)
        print("✓ Visual check complete!")
        print("="*70)
        print("\nKey observations:")
        print("  - Red boxes = Fovea (Class 1)")
        print("  - Blue boxes = SCR (Class 2)")
        print("  - Boxes are in pixel coordinates [x1, y1, x2, y2]")
        print("  - All coordinates should be within image boundaries")
        
    except Exception as e:
        print(f"\n✗ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
