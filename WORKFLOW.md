# RCNN-OCT Training and Inference Workflow

## Overview
This document describes the complete workflow for training and evaluating the Faster R-CNN model on OCT B-scan data.

## Step 1: Generate Data Splits

First, create reproducible train/val/test splits (80%/10%/10%):

```bash
python split.py \
  --data-root /home/suraj/Data/Nemours/pickle \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42 \
  --output splits.json
```

**Output**: `splits.json` containing file paths for each split

**Options**:
- `--data-root`: Directory containing `.pkl` files
- `--train-ratio`: Fraction for training (default: 0.8)
- `--val-ratio`: Fraction for validation (default: 0.1)
- `--test-ratio`: Fraction for testing (default: 0.1)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output`: Output JSON file path (default: splits.json)

## Step 2: Train Model

Train using the predefined splits:

```bash
python train.py \
  --splits-file splits.json \
  --epochs 50 \
  --batch-size 4 \
  --output-dir checkpoints \
  --filter-empty 0.0 \
  --score-threshold 0.05
```

**Key Options**:
- `--splits-file`: Path to splits.json (uses predefined splits)
- `--data-root`: If splits-file not provided, uses random split from this directory
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size (reduce if OOM errors)
- `--filter-empty`: Fraction of empty images to remove from training (0.0-1.0)
- `--score-threshold`: Score threshold for validation evaluation

**Outputs**:
- `checkpoints/best_model.pth`: Best model weights
- `checkpoints/training_results.json`: Training history and metrics

### Training Without Splits File (Legacy Mode)
If you don't have `splits.json`, the script will use random splitting:

```bash
python train.py \
  --data-root /home/suraj/Data/Nemours/pickle \
  --val-ratio 0.2 \
  --seed 42 \
  --epochs 50
```

## Step 3: Run Batch Inference

Evaluate on the test set with comprehensive metrics:

```bash
python batch_inference.py \
  --checkpoint checkpoints/best_model.pth \
  --splits-file splits.json \
  --output-dir inference \
  --score-threshold 0.5 \
  --visualize-samples 10
```

**Options**:
- `--checkpoint`: Path to trained model `.pth` file
- `--splits-file`: Path to splits.json (uses test split)
- `--output-dir`: Directory for results (default: inference/)
- `--score-threshold`: Confidence threshold for predictions (default: 0.5)
- `--visualize-samples`: Number of samples to visualize (default: 10)

**Outputs**:
- `inference/test_metrics_YYYYMMDD_HHMMSS.json`: Comprehensive metrics
- `inference/visualizations/`: Side-by-side GT vs Prediction images

### Metrics Calculated
- **Precision, Recall, F1**: At IoU=0.5
- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.5:0.95**: COCO-style mAP (average over IoU 0.5 to 0.95)
- **TP, FP, FN, TN**: Confusion matrix components

## Step 4: Single Sample Inference (Optional)

For quick testing on individual samples:

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --sample /path/to/sample.pkl \
  --score-threshold 0.5 \
  --save-path results/prediction.png
```

## Memory Management for Large GPUs

If running on A100 with MIG or encountering OOM errors:

1. **Reduce batch size**: Use `--batch-size 2` or `--batch-size 1`
2. **Set environment variable**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   python train.py ...
   ```
3. **Check GPU partitioning**:
   ```bash
   nvidia-smi
   nvidia-smi mig -lgi  # Check MIG configuration
   ```

## File Structure

```
RCNN-OCT/
├── split.py              # Generate train/val/test splits
├── train.py              # Training script
├── batch_inference.py    # Batch inference with metrics
├── inference.py          # Single sample inference
├── dataset.py            # Dataset loading utilities
├── model.py              # Model architecture
├── splits.json           # Data splits (generated)
├── checkpoints/          # Model checkpoints and training logs
│   ├── best_model.pth
│   └── training_results.json
└── inference/            # Inference results
    ├── test_metrics_*.json
    └── visualizations/
```

## Example Complete Workflow

```bash
# 1. Generate splits
python split.py --data-root /home/suraj/Data/Nemours/pickle --output splits.json

# 2. Train model
python train.py --splits-file splits.json --epochs 50 --batch-size 4 --output-dir checkpoints

# 3. Evaluate on test set
python batch_inference.py \
  --checkpoint checkpoints/best_model.pth \
  --splits-file splits.json \
  --output-dir inference \
  --visualize-samples 20
```

## Label Mapping

The dataset uses the following label encoding:
- **Label 0**: Fovea (anatomical landmark)
- **Label 1**: SCR (Sub-Clinical Retinal progression region)
- **Label 2**: No boxes (filtered out during training)

Background class is automatically assigned class ID 0 in the model, and the above labels are remapped to contiguous class IDs starting from 1.

## Tips

1. **Reproducibility**: Always use the same `--seed` value and `splits.json` file
2. **Hyperparameter tuning**: Adjust `--score-threshold` based on precision/recall trade-off
3. **Filtering empty images**: Use `--filter-empty 0.5` to remove 50% of images without boxes
4. **Quick testing**: Use `--max-samples 100` in train.py for rapid prototyping
5. **Visualization**: Increase `--visualize-samples` to inspect more predictions

## Troubleshooting

### Out of Memory Error
- Reduce `--batch-size` to 2 or 1
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Check if MIG is enabled: `nvidia-smi mig -lgi`

### Tensor Dimension Mismatch
- Ensure pickle files have correct format: `image`, `boxes`/`box`, `labels`/`label`
- Images should be grayscale (H, W) or (1, H, W) format

### Low Performance
- Try different `--score-threshold` values (e.g., 0.05, 0.3, 0.5)
- Check `--filter-empty` ratio - too aggressive filtering may hurt generalization
- Verify label mapping in `training_results.json`

## Performance Expectations

Typical metrics on well-annotated OCT data:
- **Precision**: 0.70-0.90
- **Recall**: 0.60-0.85
- **F1 Score**: 0.70-0.87
- **mAP@0.5**: 0.65-0.85
- **mAP@0.5:0.95**: 0.45-0.70

Training time: ~2-4 hours for 50 epochs on A100 GPU with batch_size=4
