#!/usr/bin/env python3
"""
Create a per-volume detection file system.
Instead of one massive 11GB file, create individual files per volume.
This allows loading only what you need.
"""

import pickle
import os
from pathlib import Path
import gc

def split_detection_file_by_volume(input_path, output_dir):
    """
    Split the large detection file into individual volume files.
    
    This approach:
    1. Loads the full file once (requires 11GB RAM)
    2. Saves each volume separately
    3. Allows future loading of only needed volumes
    
    WARNING: This still requires loading the full file once.
    If this fails, we need a different approach.
    """
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print("\n" + "="*60)
    print("ATTEMPTING TO LOAD FULL FILE")
    print("="*60)
    print("This requires ~11GB RAM and may fail if memory is insufficient.")
    print("If this crashes, you'll need to:")
    print("  1. Stop the RT-DETR training (frees 5.7 GB)")
    print("  2. Close other applications")
    print("  3. Try again")
    print("\nLoading...")
    
    try:
        with open(input_path, 'rb') as f:
            detection_data = pickle.load(f)
        
        print(f"✓ Successfully loaded {len(detection_data)} volumes")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each volume separately
        print(f"\nSaving individual volume files to: {output_dir}")
        
        for i, (vol_id, vol_data) in enumerate(detection_data.items(), 1):
            output_file = os.path.join(output_dir, f"{vol_id}.pkl")
            
            with open(output_file, 'wb') as f:
                pickle.dump(vol_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if i % 10 == 0:
                print(f"  Saved {i}/{len(detection_data)} volumes...")
        
        print(f"\n✓ Successfully saved {len(detection_data)} volume files")
        
        # Calculate total size
        total_size = sum(os.path.getsize(os.path.join(output_dir, f)) 
                        for f in os.listdir(output_dir))
        total_size_gb = total_size / (1024**3)
        
        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"{'='*60}")
        print(f"Total size: {total_size_gb:.2f} GB")
        print(f"Files created: {len(detection_data)}")
        print(f"Location: {output_dir}")
        print(f"{'='*60}")
        
        # Create a helper function file
        helper_file = os.path.join(output_dir, 'load_volume.py')
        with open(helper_file, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""Helper function to load individual volume detections."""

import pickle
import os

def load_volume_detections(volume_id, detections_dir=None):
    """
    Load detection data for a specific volume.
    
    Args:
        volume_id: Volume ID (e.g., '185_R_3')
        detections_dir: Directory containing volume files
        
    Returns:
        List of detection dictionaries for the volume
    """
    if detections_dir is None:
        detections_dir = os.path.dirname(__file__)
    
    filepath = os.path.join(detections_dir, f"{volume_id}.pkl")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No detection file found for volume: {volume_id}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Example usage
    vol_id = "185_R_3"
    detections = load_volume_detections(vol_id)
    print(f"Loaded {len(detections)} B-scans for volume {vol_id}")
''')
        
        print(f"\nHelper function created: {helper_file}")
        print("\nUsage in your notebook:")
        print(f"  from {output_dir}/load_volume import load_volume_detections")
        print(f"  detections = load_volume_detections('185_R_3')")
        
        return True
        
    except MemoryError:
        print("\n" + "="*60)
        print("ERROR: OUT OF MEMORY!")
        print("="*60)
        print("\nThe system doesn't have enough RAM to load the 11GB file.")
        print("\nCurrent memory usage:")
        import subprocess
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        print(result.stdout)
        
        print("\nRECOMMENDATIONS:")
        print("1. Stop the RT-DETR training process:")
        print("   kill 1409752")
        print("   (This will free up 5.7 GB RAM)")
        print("\n2. Close other applications")
        print("\n3. Try running this script again")
        
        return False
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    input_path = '/home/suraj/Git/RCNN-OCT/nemours_inference/full_dataset_inference.pkl'
    output_dir = '/home/suraj/Git/RCNN-OCT/nemours_inference/volumes'
    
    success = split_detection_file_by_volume(input_path, output_dir)
    
    if not success:
        print("\n" + "="*60)
        print("ALTERNATIVE SOLUTION")
        print("="*60)
        print("\nIf you can't free up enough memory, we can:")
        print("1. Use the original inference script to regenerate")
        print("   detections on-demand for specific volumes")
        print("2. Modify the inference script to save per-volume")
        print("   files instead of one large file")
        print("="*60)
