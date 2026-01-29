# Nemours Dataset Visualization with NMS

This document describes how to visualize model predictions on the Nemours dataset with Non-Maximum Suppression (NMS) to handle overlapping predictions.

## Overview

Two visualization scripts are provided:

1. **`visualize_nemours_predictions.py`** - Uses pre-computed inference results from pickle file
2. **`visualize_nemours_live.py`** - Runs inference on-the-fly (memory-efficient, recommended)

Both scripts:
- Apply **Non-Maximum Suppression (NMS)** to handle overlapping predictions
- Select the prediction with **highest confidence** when multiple predictions overlap
- Generate organized visualizations by volume
- Provide detailed statistics on predictions before/after NMS

## Non-Maximum Suppression (NMS)

NMS removes redundant overlapping predictions:
- When multiple bounding boxes overlap (IoU > threshold), keep only the one with highest confidence
- Boxes of different classes are not suppressed against each other
- Default IoU threshold: 0.5 (configurable)

## Usage

### Option 1: Live Inference (Recommended)

This approach is memory-efficient and doesn't require pre-computed results:

```bash
python visualize_nemours_live.py \
    --checkpoint checkpoints12022205/best_model.pth \
    --hdf5-path /home/suraj/Data/Nemours/Nemours_Jing_0929.h5 \
    --output-dir nemours_visualizations_live \
    --score-threshold 0.5 \
    --nms-iou-threshold 0.5 \
    --max-viz-per-volume 10
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint (required)
- `--hdf5-path`: Path to HDF5 dataset file (required)
- `--output-dir`: Output directory for visualizations (default: `nemours_visualizations_live`)
- `--score-threshold`: Confidence threshold for predictions (default: 0.5)
- `--nms-iou-threshold`: IoU threshold for NMS (default: 0.5)
- `--num-classes`: Number of classes in model (default: 3)
- `--max-volumes`: Maximum number of volumes to process (default: all)
- `--max-viz-per-volume`: Maximum visualizations per volume (default: 10)
- `--visualize-all`: Visualize all B-scans in each volume (flag)

### Option 2: Pre-computed Results

If you have pre-computed inference results in a pickle file:

```bash
python visualize_nemours_predictions.py \
    --inference-pkl nemours_inference/full_dataset_inference.pkl \
    --hdf5-path /home/suraj/Data/Nemours/Nemours_Jing_0929.h5 \
    --output-dir nemours_visualizations \
    --nms-iou-threshold 0.5 \
    --max-viz-per-volume 10
```

**Arguments:**
- `--inference-pkl`: Path to inference results pickle file (required)
- `--hdf5-path`: Path to HDF5 dataset file (required)
- `--output-dir`: Output directory for visualizations (default: `nemours_visualizations`)
- `--nms-iou-threshold`: IoU threshold for NMS (default: 0.5)
- `--max-volumes`: Maximum number of volumes to process (default: all)
- `--max-viz-per-volume`: Maximum visualizations per volume (default: 10)
- `--visualize-all`: Visualize all B-scans in each volume (flag)

## Examples

### Process all volumes with default settings

```bash
python visualize_nemours_live.py \
    --checkpoint checkpoints12022205/best_model.pth \
    --hdf5-path /home/suraj/Data/Nemours/Nemours_Jing_0929.h5
```

### Process first 10 volumes with all B-scans visualized

```bash
python visualize_nemours_live.py \
    --checkpoint checkpoints12022205/best_model.pth \
    --hdf5-path /home/suraj/Data/Nemours/Nemours_Jing_0929.h5 \
    --max-volumes 10 \
    --visualize-all
```

### Use stricter NMS threshold (less suppression)

```bash
python visualize_nemours_live.py \
    --checkpoint checkpoints12022205/best_model.pth \
    --hdf5-path /home/suraj/Data/Nemours/Nemours_Jing_0929.h5 \
    --nms-iou-threshold 0.3
```

### Lower confidence threshold to see more predictions

```bash
python visualize_nemours_live.py \
    --checkpoint checkpoints12022205/best_model.pth \
    --hdf5-path /home/suraj/Data/Nemours/Nemours_Jing_0929.h5 \
    --score-threshold 0.3
```

## Output Structure

Visualizations are organized by volume:

```
nemours_visualizations_live/
├── 256_L_1_1/
│   ├── bscan_000.png
│   ├── bscan_003.png
│   ├── bscan_006.png
│   └── ...
├── 244_R_1/
│   ├── bscan_000.png
│   ├── bscan_003.png
│   └── ...
└── ...
```

Each visualization shows:
- B-scan image in grayscale
- Predicted bounding boxes with confidence scores
- Color coding: **Blue** = Fovea, **Red** = SCR
- Title with volume ID, B-scan index, and prediction counts

## Output Statistics

The script prints comprehensive statistics:

```
SUMMARY STATISTICS
==============================================================
Total volumes processed: 104
Total B-scans: 2540
Total B-scans visualized: 1040
B-scans with predictions: 856

Predictions before NMS: 1243
Predictions after NMS: 1089
Predictions removed by NMS: 154
Reduction: 12.4%

Per-Volume Statistics:
--------------------------------------------------------------
Volume ID                 B-scans    Viz      Pred (Before)   Pred (After)
--------------------------------------------------------------
256_L_1_1                 31         10       15              13
244_R_1                   31         10       18              16
...
```

## Visualization Color Coding

- **Blue boxes**: Fovea predictions
- **Red boxes**: SCR (Subretinal Cystoid Retinoschisis) predictions
- **Text labels**: Show class name and confidence score

## NMS Parameters

The NMS IoU threshold controls how aggressively overlapping boxes are suppressed:

- **0.3**: More aggressive (removes boxes with >30% overlap)
- **0.5**: Balanced (default, removes boxes with >50% overlap)
- **0.7**: Less aggressive (only removes boxes with >70% overlap)

Choose based on your use case:
- **Lower threshold** (0.3-0.4): Use when you want to avoid duplicate detections
- **Higher threshold** (0.6-0.7): Use when you want to keep more predictions

## Notes

- The live inference version (`visualize_nemours_live.py`) is recommended for large datasets
- Pre-computed results version requires the entire pickle file to fit in memory
- Visualizations are sampled evenly across each volume unless `--visualize-all` is used
- NMS is applied per B-scan, not across the entire volume
