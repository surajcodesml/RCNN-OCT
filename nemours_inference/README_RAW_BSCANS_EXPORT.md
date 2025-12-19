# Raw B-scan Export Documentation

## Overview
This document describes the export process of raw OCT B-scan images from the Nemours HDF5 dataset for expert annotation and model performance evaluation.

## Purpose
Raw B-scans were exported for **expert manual annotation** to create ground truth bounding boxes for Fovea and Sub-Clinical Retinal (SCR) features. These expert annotations will be used to evaluate the object detection model's performance against human expert labels.

## Source Dataset
- **File**: `/home/suraj/Data/Nemours/Nemours_Jing_0929.h5`
- **Total Volumes**: 254 volumes
- **Total B-scans**: 7,868 B-scans
- **Image Dimensions**: 496 × 768 pixels (Height × Width)
- **Format**: uint8 grayscale images

## Export Specifications

### Selected Volumes (20 volumes)
The following 20 volumes were selected for expert annotation:

1. 48_R_8
2. 242_R_1
3. 124_R_5
4. 239_R_1
5. 152_R_3_1
6. 124_L_5
7. 259_R_1
8. 30_L_5
9. 240_R_1
10. 240_L_1
11. 62_L_6
12. 22_R_5
13. 185_R_3
14. 241_R_2_1
15. 30_R_5
16. 47_R_6
17. 47_L_6
18. 86_L_6
19. 247_R_1
20. 62_R_6

### Export Details
- **Total B-scans Exported**: All B-scans from the 20 selected volumes (~31 B-scans per volume on average)
- **Image Format**: PNG (grayscale, 8-bit)
- **Naming Convention**: `{volume_id}_bscan_{bscan_index:03d}.png`
  - Example: `48_R_8_bscan_015.png` (Volume 48_R_8, B-scan index 15)
- **No Pre-processing**: Images are exported exactly as stored in HDF5 (no normalization, resizing, or augmentation)
- **No Annotations**: Raw images contain no bounding boxes, labels, or overlays

## Directory Structure

```
nemours_inference/
├── selected_20_volumes_raw_images/          # Raw B-scan images
│   ├── 48_R_8_bscan_000.png
│   ├── 48_R_8_bscan_001.png
│   ├── ...
│   ├── 62_R_6_bscan_030.png
│   └── selected_volumes_metadata.json       # Metadata file
└── selected_20_volumes_detections/          # Model predictions (for reference)
    ├── 48_R_8_bscan_000.png
    └── ...
```

## Metadata File Structure

The `selected_volumes_metadata.json` file contains comprehensive information about each exported B-scan:

```json
{
  "total_samples": 620,
  "num_target_volumes": 20,
  "volumes_found": ["48_R_8", "242_R_1", ...],
  "volumes_not_found": [],
  "target_volumes": ["48_R_8", "242_R_1", ...],
  "source_hdf5": "/home/suraj/Data/Nemours/Nemours_Jing_0929.h5",
  "image_dimensions": "496x768",
  "samples": [
    {
      "index": 0,
      "filename": "48_R_8_bscan_015.png",
      "volume_id": "48_R_8",
      "bscan_index": 15,
      "global_index": 1523,
      "image_shape": [496, 768],
      "num_detections": 2,
      "detection_labels": [1, 2],
      "detection_scores": [0.95, 0.87]
    },
    ...
  ]
}
```

### Metadata Fields Explanation

#### Top-Level Fields
- **total_samples**: Total number of B-scans exported
- **num_target_volumes**: Number of volumes selected (20)
- **volumes_found**: List of volumes successfully found in the dataset
- **volumes_not_found**: List of requested volumes not found (if any)
- **target_volumes**: Original list of requested volume IDs
- **source_hdf5**: Full path to the source HDF5 file
- **image_dimensions**: Image dimensions in format "height×width"

#### Per-Sample Fields
- **index**: Sequential index in the exported dataset (0 to N-1)
- **filename**: PNG filename of the exported B-scan
- **volume_id**: Volume identifier (patient/eye/visit)
- **bscan_index**: B-scan position within the volume (0-30)
- **global_index**: Index in the original HDF5 dataset (for re-extraction)
- **image_shape**: [height, width] in pixels
- **num_detections**: Number of objects detected by the model
- **detection_labels**: Label IDs detected (1=Fovea, 2=SCR)
- **detection_scores**: Confidence scores for each detection (0-1)

## Model Predictions (For Reference)

The model's predictions are saved separately in `selected_20_volumes_detections/` directory. These visualizations include:
- Bounding boxes overlaid on images
- Class labels (Fovea = green, SCR = red)
- Confidence scores
- Detection counts

**Note**: These are NOT ground truth annotations. They are the model's predictions and should only be used as reference during expert annotation.

## Expert Annotation Guidelines

### Annotation Task
Experts should annotate the raw B-scan images (in `selected_20_volumes_raw_images/`) by drawing bounding boxes around:

1. **Fovea (Class 1)**: The central depression of the retina
   - Typically appears as a dip or pit in the central region
   - Usually one fovea per B-scan (if present)

2. **Sub-Clinical Retinal Features / SCR (Class 2)**: Abnormal retinal features
   - May include drusen, fluid accumulation, or structural changes
   - Can have multiple instances per B-scan

### Annotation Format
Expert annotations should be saved in a format compatible with object detection evaluation (e.g., COCO JSON, Pascal VOC XML, or YOLO txt format) with the following information:
- Image filename
- Bounding box coordinates (x1, y1, x2, y2) or (x, y, width, height)
- Class label (1 or 2)
- Confidence/certainty (optional)

### Recommended Annotation Order
1. Review the model predictions in `selected_20_volumes_detections/` to understand the task
2. Annotate raw images in `selected_20_volumes_raw_images/` independently
3. Use metadata file to track progress and identify problematic cases

## Model Evaluation Plan

Once expert annotations are complete, model performance will be evaluated using:

### Metrics
1. **Detection Metrics**:
   - Precision, Recall, F1-Score
   - Mean Average Precision (mAP) at various IoU thresholds (0.5, 0.75, 0.5:0.95)
   - Confusion matrix (Fovea vs SCR vs Background)

2. **Localization Metrics**:
   - Intersection over Union (IoU)
   - Boundary accuracy (distance between predicted and ground truth box edges)

3. **Per-Volume Analysis**:
   - Performance stratification by volume
   - Analysis of challenging cases (low confidence, missed detections, false positives)

### Evaluation Process
1. Load expert annotations (ground truth)
2. Load model predictions from `selected_volumes_metadata.json`
3. Match predictions to ground truth using IoU threshold
4. Calculate metrics per class and overall
5. Generate performance reports and visualizations

## Re-extraction from HDF5

If additional B-scans need to be extracted, use the `global_index` field from the metadata:

```python
import h5py

with h5py.File('/home/suraj/Data/Nemours/Nemours_Jing_0929.h5', 'r') as f:
    images = f['images'][:]
    
    # Extract specific B-scan using global_index
    global_idx = 1523  # From metadata
    bscan_image = images[global_idx]
```

## Export Script

The export was performed using the Jupyter notebook:
- **Notebook**: `/home/suraj/Git/RCNN-OCT/nemours_inference.ipynb`
- **Cell**: Section 5 - "Task 2: Sample Images from Selected Volumes"
- **Date**: December 19, 2025
- **Model**: `checkpoints12022205/best_model.pth`
- **Score Threshold**: 0.5 (for initial detections)

## Contact & Questions

For questions about the exported data, annotation process, or evaluation methodology, please refer to:
- Main project documentation: `/home/suraj/Git/RCNN-OCT/README.md`
- Inference workflow: `/home/suraj/Git/RCNN-OCT/WORKFLOW.md`
- Dataset documentation: `/home/suraj/Data/Nemours/`

---

**Last Updated**: December 19, 2025  
**Export Version**: 1.0  
**Status**: Ready for expert annotation
