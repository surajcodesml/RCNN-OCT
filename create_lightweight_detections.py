#!/usr/bin/env python3
"""
Create a lightweight version of the detection file containing only essential data.
This reduces the 11 GB file to a much smaller size by removing unnecessary data.
"""

import pickle
import numpy as np
from tqdm import tqdm
import sys

def get_file_size_gb(filepath):
    """Get file size in GB."""
    import os
    size_bytes = os.path.getsize(filepath)
    return size_bytes / (1024**3)

def create_lightweight_detections(input_path, output_path):
    """
    Extract only essential detection data:
    - bscan_idx
    - pred_boxes
    - pred_labels  
    - pred_scores
    
    Remove any images, features, or other large intermediate data.
    """
    print(f"Loading detection file: {input_path}")
    print(f"Original file size: {get_file_size_gb(input_path):.2f} GB")
    print("\nThis may take a few minutes and use significant memory...")
    print("If this crashes, we'll need to use a streaming approach.\n")
    
    try:
        with open(input_path, 'rb') as f:
            detection_data = pickle.load(f)
        
        print(f"Loaded {len(detection_data)} volumes")
        
        # Create lightweight version
        lightweight_detections = {}
        total_bscans = sum(len(bscans) for bscans in detection_data.values())
        
        print(f"\nProcessing {total_bscans} B-scans across {len(detection_data)} volumes...")
        
        with tqdm(total=len(detection_data), desc="Processing volumes") as pbar:
            for vol_id, bscans in detection_data.items():
                lightweight_detections[vol_id] = []
                
                for bscan_data in bscans:
                    # Extract only essential fields
                    essential_data = {
                        'bscan_idx': bscan_data['bscan_idx'],
                        'pred_boxes': bscan_data['pred_boxes'],
                        'pred_labels': bscan_data['pred_labels'],
                        'pred_scores': bscan_data['pred_scores']
                    }
                    
                    lightweight_detections[vol_id].append(essential_data)
                
                pbar.update(1)
        
        # Save lightweight version
        print(f"\nSaving lightweight detections to: {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(lightweight_detections, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        output_size = get_file_size_gb(output_path)
        original_size = get_file_size_gb(input_path)
        reduction = ((original_size - output_size) / original_size) * 100
        
        print(f"\n{'='*60}")
        print(f"SUCCESS!")
        print(f"{'='*60}")
        print(f"Original file size:    {original_size:.2f} GB")
        print(f"Lightweight file size: {output_size:.2f} GB")
        print(f"Size reduction:        {reduction:.1f}%")
        print(f"{'='*60}")
        
        return True
        
    except MemoryError:
        print("\n" + "="*60)
        print("ERROR: Out of memory!")
        print("="*60)
        print("The file is too large to load all at once.")
        print("\nRecommendations:")
        print("1. Stop the RT-DETR training process to free up 5.7 GB RAM")
        print("2. Close other applications")
        print("3. Try again")
        print("\nAlternatively, we can create a streaming version that")
        print("processes the file in chunks without loading it all.")
        print("="*60)
        return False
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    input_path = '/home/suraj/Git/RCNN-OCT/nemours_inference/full_dataset_inference.pkl'
    output_path = '/home/suraj/Git/RCNN-OCT/nemours_inference/full_dataset_inference_lite.pkl'
    
    print("="*60)
    print("Creating Lightweight Detection File")
    print("="*60)
    
    success = create_lightweight_detections(input_path, output_path)
    
    if success:
        print("\nYou can now use the lightweight file in your notebook:")
        print(f"  detection_path = '{output_path}'")
    else:
        print("\nFailed to create lightweight file.")
        print("Please free up memory and try again.")
        sys.exit(1)
