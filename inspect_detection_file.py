#!/usr/bin/env python3
"""
Inspect the detection pickle file to understand its structure and size.
This script loads only a small sample to avoid memory issues.
"""

import pickle
import numpy as np
import sys

def inspect_detection_file(filepath):
    """
    Inspect the structure of the detection file without loading everything.
    """
    print(f"Inspecting: {filepath}\n")
    
    try:
        with open(filepath, 'rb') as f:
            # Try to load just the first part
            print("Loading detection data (this may take a moment)...")
            detection_data = pickle.load(f)
        
        print(f"✓ Successfully loaded detection data")
        print(f"\nTop-level structure:")
        print(f"  Type: {type(detection_data)}")
        print(f"  Number of volumes: {len(detection_data)}")
        
        # Get first volume
        first_vol_id = list(detection_data.keys())[0]
        first_vol_data = detection_data[first_vol_id]
        
        print(f"\nFirst volume: {first_vol_id}")
        print(f"  Type: {type(first_vol_data)}")
        print(f"  Number of B-scans: {len(first_vol_data)}")
        
        # Get first B-scan
        if len(first_vol_data) > 0:
            first_bscan = first_vol_data[0]
            print(f"\nFirst B-scan structure:")
            print(f"  Type: {type(first_bscan)}")
            print(f"  Keys: {list(first_bscan.keys())}")
            
            # Inspect each field
            print(f"\nField details:")
            for key, value in first_bscan.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}:")
                    print(f"    Type: numpy.ndarray")
                    print(f"    Shape: {value.shape}")
                    print(f"    Dtype: {value.dtype}")
                    print(f"    Size: {value.nbytes / (1024**2):.2f} MB")
                else:
                    print(f"  {key}: {type(value)} = {value}")
        
        # Calculate total size estimate
        print(f"\n{'='*60}")
        print("Estimating total data size...")
        print(f"{'='*60}")
        
        total_size_mb = 0
        sample_count = 0
        
        for vol_id in list(detection_data.keys())[:5]:  # Sample first 5 volumes
            for bscan in detection_data[vol_id]:
                for key, value in bscan.items():
                    if isinstance(value, np.ndarray):
                        total_size_mb += value.nbytes / (1024**2)
                sample_count += 1
        
        avg_size_per_bscan = total_size_mb / sample_count if sample_count > 0 else 0
        total_bscans = sum(len(bscans) for bscans in detection_data.values())
        estimated_total_gb = (avg_size_per_bscan * total_bscans) / 1024
        
        print(f"Sample size: {sample_count} B-scans from 5 volumes")
        print(f"Average size per B-scan: {avg_size_per_bscan:.2f} MB")
        print(f"Total B-scans in dataset: {total_bscans}")
        print(f"Estimated total size: {estimated_total_gb:.2f} GB")
        
        # Check what fields are taking up space
        print(f"\n{'='*60}")
        print("Analyzing field sizes (from first B-scan)...")
        print(f"{'='*60}")
        
        if len(first_vol_data) > 0:
            field_sizes = {}
            for key, value in first_bscan.items():
                if isinstance(value, np.ndarray):
                    field_sizes[key] = value.nbytes / (1024**2)
            
            # Sort by size
            sorted_fields = sorted(field_sizes.items(), key=lambda x: x[1], reverse=True)
            
            for key, size_mb in sorted_fields:
                print(f"  {key:20s}: {size_mb:8.2f} MB")
        
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS:")
        print(f"{'='*60}")
        
        # Identify unnecessary fields
        essential_fields = {'bscan_idx', 'pred_boxes', 'pred_labels', 'pred_scores'}
        if len(first_vol_data) > 0:
            actual_fields = set(first_bscan.keys())
            unnecessary_fields = actual_fields - essential_fields
            
            if unnecessary_fields:
                print(f"\nUnnecessary fields detected (can be removed):")
                for field in unnecessary_fields:
                    if isinstance(first_bscan[field], np.ndarray):
                        size_mb = first_bscan[field].nbytes / (1024**2)
                        print(f"  - {field} ({size_mb:.2f} MB per B-scan)")
                
                print(f"\nRemoving these fields could reduce file size significantly!")
            else:
                print(f"\nFile contains only essential fields.")
                print(f"The large size is due to the number of detections.")
        
    except MemoryError:
        print("\n" + "="*60)
        print("ERROR: Out of memory while inspecting file!")
        print("="*60)
        print("The file is too large to load even for inspection.")
        print("\nCurrent memory status:")
        import subprocess
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        print(result.stdout)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    filepath = '/home/suraj/Git/RCNN-OCT/nemours_inference/full_dataset_inference.pkl'
    inspect_detection_file(filepath)
