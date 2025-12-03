# Configuration Guide for RCNN-OCT

## Overview
All scripts in this project use a centralized YAML configuration file (`config.yaml`) to manage paths and hyperparameters. This eliminates the need to manually change arguments across multiple scripts.

## Quick Start

### 1. Edit config.yaml
Update the data path in `config.yaml`:
```yaml
data:
  root: "/your/path/to/pickle/files"
```

### 2. Run the complete workflow
```bash
bash run_workflow.sh
```

This will:
1. Generate train/val/test splits
2. Train the model
3. Run inference on test set

## Configuration File Structure

### Data Paths
```yaml
data:
  root: "/home/suraj/Data/Nemours/pickle"  # Root directory with .pkl files
  splits_file: "splits.json"                # Where to save/load splits
```

### Training Hyperparameters
```yaml
training:
  epochs: 50                    # Number of training epochs
  batch_size: 4                 # Training batch size
  learning_rate: 0.005          # Initial learning rate
  momentum: 0.9                 # SGD momentum
  weight_decay: 0.0005          # L2 regularization
  lr_scheduler_step_size: 5     # Epochs between LR decay
  lr_scheduler_gamma: 0.1       # LR decay factor
  num_workers: 4                # DataLoader workers
  patience: 7                   # Early stopping patience
  filter_empty_ratio: 0.0       # Fraction of empty images to remove (0.0-1.0)
  score_threshold: 0.05         # Validation score threshold
```

### Data Splitting
```yaml
splitting:
  train_ratio: 0.8    # 80% for training
  val_ratio: 0.1      # 10% for validation
  test_ratio: 0.1     # 10% for testing
  seed: 42            # Random seed for reproducibility
```

### Inference Settings
```yaml
inference:
  score_threshold: 0.5      # Higher threshold for final predictions
  visualize_samples: 10     # Number of samples to visualize
```

### Output Directories
```yaml
output:
  checkpoints_dir: "checkpoints"   # Model checkpoints
  inference_dir: "inference"       # Inference results
  splits_file: "splits.json"       # Train/val/test splits
```

## Usage Examples

### Generate Splits
Using config file:
```bash
python split.py --config config.yaml
```

Override specific values:
```bash
python split.py --config config.yaml --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

### Train Model
Using config file:
```bash
python train.py --config config.yaml
```

Override epochs and batch size:
```bash
python train.py --config config.yaml --epochs 100 --batch-size 8
```

### Run Inference
Using config file:
```bash
python batch_inference.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

Override score threshold:
```bash
python batch_inference.py \
  --config config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --score-threshold 0.7 \
  --visualize-samples 20
```

## Command Line Overrides

All scripts support command-line arguments that **override** the config file:

### split.py
- `--config`: Path to config YAML (default: config.yaml)
- `--data-root`: Override data root path
- `--train-ratio`: Override train ratio
- `--val-ratio`: Override validation ratio
- `--test-ratio`: Override test ratio
- `--seed`: Override random seed
- `--output`: Override output splits file path

### train.py
- `--config`: Path to config YAML (default: config.yaml)
- `--splits-file`: Override splits file path
- `--epochs`: Override number of epochs
- `--batch-size`: Override batch size
- `--output-dir`: Override output directory

### batch_inference.py
- `--config`: Path to config YAML (default: config.yaml)
- `--checkpoint`: Path to model checkpoint (required)
- `--splits-file`: Override splits file path
- `--output-dir`: Override output directory
- `--score-threshold`: Override score threshold
- `--visualize-samples`: Override number of visualizations

## Multiple Configurations

You can maintain multiple config files for different experiments:

```bash
# Training with aggressive filtering
python train.py --config config_filtered.yaml

# Training with original data
python train.py --config config_original.yaml

# Quick test run
python train.py --config config_test.yaml
```

Example `config_test.yaml`:
```yaml
training:
  epochs: 5
  batch_size: 2
  
splitting:
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
```

## Best Practices

1. **Use config.yaml for production runs**: Set all hyperparameters once, run consistently
2. **Override for experiments**: Use command-line args for quick parameter sweeps
3. **Version control configs**: Commit config files alongside code
4. **Document changes**: Add comments in YAML for non-standard settings
5. **Keep splits consistent**: Always use the same `splits.json` for a project

## Troubleshooting

### FileNotFoundError: splits.json not found
Run `python split.py --config config.yaml` first

### ModuleNotFoundError: No module named 'yaml'
Install PyYAML: `pip install pyyaml` or `conda install pyyaml`

### Invalid config file
Ensure proper YAML syntax:
- Use spaces, not tabs
- Maintain proper indentation
- Quote paths with special characters

### Different paths on different machines
Use environment variables or create machine-specific configs:
```yaml
data:
  root: "${DATA_ROOT:-/default/path}"
```

## Related Files

- `config.yaml` - Main configuration file
- `split.py` - Generate data splits
- `train.py` - Training script
- `batch_inference.py` - Batch inference with metrics
- `run_workflow.sh` - Complete pipeline automation
- `WORKFLOW.md` - Detailed workflow documentation
