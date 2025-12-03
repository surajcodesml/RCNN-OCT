# RCNN-OCT Training and Inference Workflow

## Overview
This document describes the complete workflow for training and evaluating the Faster R-CNN model on OCT B-scan data.

**NEW**: All scripts now use a centralized `config.yaml` file. See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for details.

## Quick Start

1. **Edit config.yaml** with your data path
2. **Run the workflow**:
   ```bash
   bash run_workflow.sh
   ```

That's it! The script will generate splits, train the model, and run inference.

## Step-by-Step Guide

## Step 1: Generate Data Splits

Create reproducible train/val/test splits (80%/10%/10%):

```bash
python split.py --config config.yaml
```

**Output**: `splits.json` containing file paths for each split

**Manual override example**:
```bash
python split.py \
  --config config.yaml \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

## Step 2: Train Model

Train using the config file:

```bash
python train.py --config config.yaml
```

**Override examples**:
```bash
# Quick test with fewer epochs
python train.py --config config.yaml --epochs 10 --batch-size 2

# Full training run
python train.py --config config.yaml --epochs 50 --batch-size 4
```

**Outputs**:
- `checkpoints/best_model.pth`: Best model weights
- `checkpoints/training_results.json`: Training history and metrics

## Step 3: Run Batch Inference

Evaluate on the test set:

```bash
python batch_inference.py \
  --config config.yaml \
  --checkpoint checkpoints/best_model.pth
```

**Override examples**:
```bash
# Higher confidence threshold
python batch_inference.py \
  --config config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --score-threshold 0.7 \
  --visualize-samples 20
```

**Outputs**:
- `inference/test_metrics_YYYYMMDD_HHMMSS.json`: Comprehensive metrics
- `inference/visualizations/`: Side-by-side GT vs Prediction images

### Metrics Calculated
- **Precision, Recall, F1**: At IoU=0.5
- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.5:0.95**: COCO-style mAP (average over IoU 0.5 to 0.95)
- **TP, FP, FN, TN**: Confusion matrix components

## Configuration Management

### Default Config (config.yaml)
All hyperparameters and paths are centralized:
```yaml
data:
  root: "/home/suraj/Data/Nemours/pickle"
  splits_file: "splits.json"

training:
  epochs: 50
  batch_size: 4
  learning_rate: 0.005
  # ... more settings
```

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for complete configuration documentation.

### Command-Line Overrides
All scripts accept `--config config.yaml` and support overriding specific values:
```bash
python train.py --config config.yaml --epochs 100 --batch-size 8
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
