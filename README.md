# RCNN-OCT: Retinal OCT Object Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Faster R-CNN pipeline for automated detection of anatomical landmarks and pathological features in retinal Optical Coherence Tomography (OCT) B-scan images.

## 🎯 Overview

This repository implements an end-to-end deep learning pipeline for object detection in OCT B-scans using PyTorch and torchvision's Faster R-CNN with ResNet-50-FPN backbone. The system is designed for:

- **Fovea Detection**: Localization of the foveal center (anatomical landmark)
- **Sub-Clinical Retinal (SCR) Detection**: Identification of early-stage pathological changes

### Key Features

✅ **Centralized Configuration**: Single YAML file for all hyperparameters and paths  
✅ **Reproducible Splits**: Deterministic train/validation/test splitting with seed control  
✅ **Comprehensive Metrics**: Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, and confusion matrix  
✅ **GPU/ROCm Support**: Compatible with NVIDIA CUDA and AMD ROCm  
✅ **Color-Coded Visualizations**: Intuitive green (Fovea) and red (SCR) bounding boxes  
✅ **Early Stopping**: Automatic training termination based on validation performance  

---

## 📋 Table of Contents

- [Installation](#installation)
- [Data Format](#data-format)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
  - [1. Generate Data Splits](#1-generate-data-splits)
  - [2. Train Model](#2-train-model)
  - [3. Run Inference](#3-run-inference)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualization](#visualization)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (for NVIDIA GPUs) or ROCm 5.4+ (for AMD GPUs)
- 8GB+ GPU memory recommended

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/surajcodesml/RCNN-OCT.git
cd RCNN-OCT
```

2. **Create virtual environment**
```bash
conda create -n oct python=3.10
conda activate oct
```

3. **Install dependencies**
```bash
# For NVIDIA GPUs (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For AMD GPUs (ROCm)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2

# Install additional requirements
pip install pyyaml numpy matplotlib tqdm
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 📊 Data Format

### Input Format

Each OCT B-scan is stored as a `.pkl` (pickle) file with the following structure:

```python
{
    "image": torch.Tensor,  # Shape: (3, H, W) or (H, W), values in [0, 1]
    "box": np.ndarray,      # Shape: (N, 4), YOLO format [x_center, y_center, width, height]
    "label": np.ndarray     # Shape: (N,), integer class labels
}
```

### Label Encoding

| Original Label | Description | Model Class | Visualization Color |
|----------------|-------------|-------------|---------------------|
| 0 | Fovea (anatomical landmark) | 1 | 🟢 Green |
| 1 | SCR (Sub-Clinical Retinal) | 2 | 🔴 Red |
| 2 | No boxes (ignored) | - | - |

**Note**: The model automatically assigns class 0 to background. Original labels are remapped to contiguous class IDs (1, 2) during training.

### File Naming Convention

```
patientID_LorR_instance_instanceNumber_bscanNumber.pkl
```

**Example**: `6_R_1_1001.pkl`
- Patient ID: 6
- Eye: Right (R) or Left (L)
- Instance: 1
- B-scan number: 1001

### Coordinate System

- **Input**: YOLO format with normalized coordinates [0, 1]
  - `[x_center, y_center, width, height]`
- **Internal**: Automatically converted to corner format for Faster R-CNN
  - `[x1, y1, x2, y2]` in pixel coordinates

---

## ⚡ Quick Start

### 1. Configure Your Data Path

Edit `config.yaml`:
```yaml
data:
  root: "/path/to/your/pickle/files"
```

### 2. Run Complete Pipeline

```bash
bash run_workflow.sh
```

This single command will:
1. Generate train/val/test splits (80/10/10)
2. Train the Faster R-CNN model
3. Evaluate on test set with comprehensive metrics
4. Generate visualizations

### 3. View Results

```
├── splits.json                          # Data splits
├── checkpoints/
│   ├── best_model.pth                  # Trained model weights
│   └── training_results.json           # Training history
└── inference/
    ├── test_metrics_YYYYMMDD_HHMMSS.json  # Evaluation metrics
    └── visualizations/                     # Side-by-side comparisons
```

---

## 📁 Project Structure

```
RCNN-OCT/
├── config.yaml                  # Central configuration file
├── split.py                     # Generate train/val/test splits
├── train.py                     # Training script with validation
├── batch_inference.py           # Batch inference with metrics
├── inference.py                 # Single-sample inference utility
├── dataset.py                   # Dataset loader and utilities
├── model.py                     # Faster R-CNN model builder
├── run_workflow.sh              # Complete pipeline automation
│
├── docs/
│   ├── CONFIG_GUIDE.md          # Configuration documentation
│   └── WORKFLOW.md              # Detailed usage guide
│
├── checkpoints/                 # Model weights and training logs
│   ├── best_model.pth
│   └── training_results.json
│
├── inference/                   # Inference results
│   ├── test_metrics_*.json
│   └── visualizations/
│
└── splits.json                  # Train/val/test file assignments
```

---

## ⚙️ Configuration

All hyperparameters and paths are managed through `config.yaml`:

```yaml
# Data paths
data:
  root: "/home/suraj/Data/Nemours/pickle"
  splits_file: "splits.json"

# Training hyperparameters
training:
  epochs: 50
  batch_size: 4
  learning_rate: 0.005
  momentum: 0.9
  weight_decay: 0.0005
  patience: 7                    # Early stopping patience
  filter_empty_ratio: 0.0        # Remove empty images (0.0-1.0)
  score_threshold: 0.05          # Validation threshold

# Data splitting
splitting:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  seed: 42

# Inference settings
inference:
  score_threshold: 0.5           # Prediction confidence threshold
  visualize_samples: 10          # Number of visualizations
```

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for complete documentation.

---

## 🔧 Usage

### 1. Generate Data Splits

Create reproducible train/validation/test splits:

```bash
python split.py --config config.yaml
```

**Custom splits**:
```bash
python split.py \
  --config config.yaml \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

**Output**: `splits.json` containing file paths for each split

### 2. Train Model

Train using configuration file:

```bash
python train.py --config config.yaml
```

**Override hyperparameters**:
```bash
# Quick test (10 epochs, small batch)
python train.py --config config.yaml --epochs 10 --batch-size 2

# Production training (50 epochs, larger batch)
python train.py --config config.yaml --epochs 50 --batch-size 4
```

**Training outputs**:
- `checkpoints/best_model.pth`: Model with best validation F1 score
- `checkpoints/training_results.json`: Complete training history

**Training logs include**:
- Loss components (classifier, box regression, objectness, RPN)
- Validation metrics (Precision, Recall, F1, mAP@0.5)
- True Positives, False Positives, False Negatives
- Learning rate schedule

### 3. Run Inference

Evaluate on test set with comprehensive metrics:

```bash
python batch_inference.py \
  --config config.yaml \
  --checkpoint checkpoints/best_model.pth
```

**Custom inference**:
```bash
python batch_inference.py \
  --config config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --score-threshold 0.7 \
  --visualize-samples 20
```

**Inference outputs**:
- `inference/test_metrics_*.json`: Detailed metrics
- `inference/visualizations/`: Ground truth vs predictions

### Single Sample Inference

For quick testing on individual images:

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --sample /path/to/sample.pkl \
  --score-threshold 0.5 \
  --save-path result.png
```

---

## 🏗️ Model Architecture

### Faster R-CNN with ResNet-50-FPN

- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Pretrained**: ImageNet weights (transfer learning)
- **Detection Head**: Custom FastRCNNPredictor for 3 classes (Background, Fovea, SCR)
- **Input Size**: Variable (automatically resized by model)
- **Typical Input**: 256×576 pixels (grayscale converted to 3-channel)

### Training Strategy

- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: 0.005 with step decay (γ=0.1 every 5 epochs)
- **Loss Function**: Multi-task loss (classification + box regression + RPN)
- **Early Stopping**: Patience of 7 epochs based on validation F1
- **Validation**: IoU threshold of 0.5 for positive matches

---

## 📈 Evaluation Metrics

### Detection Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Precision** | TP / (TP + FP) | >0.85 |
| **Recall** | TP / (TP + FN) | >0.80 |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | >0.82 |
| **mAP@0.5** | Mean Average Precision at IoU=0.5 | >0.75 |
| **mAP@0.5:0.95** | COCO-style mAP (average over IoU thresholds) | >0.55 |

### Confusion Matrix

- **True Positives (TP)**: Correct detections (IoU ≥ 0.5)
- **False Positives (FP)**: Incorrect detections
- **False Negatives (FN)**: Missed ground truth boxes
- **True Negatives (TN)**: Correctly predicted no objects

### Metrics Output Format

```json
{
  "test_samples": 150,
  "score_threshold": 0.5,
  "metrics_at_iou_0.5": {
    "precision": 0.8732,
    "recall": 0.8215,
    "f1": 0.8465,
    "mAP": 0.7892,
    "tp": 164,
    "fp": 24,
    "fn": 36,
    "tn": 12,
    "total_predictions": 188
  },
  "mAP@0.5:0.95": 0.5847
}
```

---

## 🎨 Visualization

### Color-Coded Bounding Boxes

- **🟢 Green**: Fovea (Label 1)
- **🔴 Red**: SCR - Sub-Clinical Retinal (Label 2)

### Visualization Output

Each visualization shows:
- **Left panel**: Ground truth annotations
- **Right panel**: Model predictions with confidence scores
- **Label format**: `Class: Confidence` (e.g., "Fovea: 0.95", "SCR: 0.87")
- **High-resolution**: 150 DPI PNG images

### Example

```
Ground Truth              Predictions
┌────────────────┐        ┌────────────────┐
│   [Green Box]  │        │   [Green Box]  │
│   GT: Fovea    │        │   Fovea: 0.95  │
│                │        │                │
│   [Red Box]    │        │   [Red Box]    │
│   GT: SCR      │        │   SCR: 0.87    │
└────────────────┘        └────────────────┘
```

---

## 🔬 Advanced Usage

### Custom Configurations

Create multiple config files for different experiments:

```bash
# Experiment with aggressive empty image filtering
python train.py --config config_filtered.yaml

# Quick validation run
python train.py --config config_debug.yaml --epochs 5
```

### Hyperparameter Tuning

Key parameters to adjust:

1. **Batch Size**: Trade-off between speed and memory
   - Small GPU (8GB): `batch_size: 2`
   - Medium GPU (16GB): `batch_size: 4`
   - Large GPU (32GB+): `batch_size: 8`

2. **Score Threshold**:
   - Training: `0.05` (low threshold for validation)
   - Inference: `0.5-0.7` (higher for production)

3. **Empty Image Filtering**:
   - `filter_empty_ratio: 0.0` (keep all)
   - `filter_empty_ratio: 0.5` (remove 50% of empty images)
   - `filter_empty_ratio: 1.0` (remove all empty images)

### Multi-GPU Training

For data parallelism, modify `train.py`:
```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

### Mixed Precision Training

For faster training with Tensor Cores:
```python
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()
# Implement in training loop
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

**Solutions**:
1. Reduce batch size: `--batch-size 2` or `--batch-size 1`
2. Enable memory optimization:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```
3. Check MIG partitioning on A100 GPUs:
   ```bash
   nvidia-smi mig -lgi
   ```

### FileNotFoundError: splits.json

**Solution**: Generate splits first
```bash
python split.py --config config.yaml
```

### ModuleNotFoundError: yaml

**Solution**: Install PyYAML
```bash
pip install pyyaml
```

### Low Performance

**Check**:
1. Score threshold (too high filters valid detections)
2. Training epochs (may need more iterations)
3. Data quality (verify annotations)
4. Class imbalance (use `filter_empty_ratio`)

### Tensor Dimension Mismatch

**Solution**: Ensure pickle files have correct format
```python
# Verify your data
import pickle
with open('sample.pkl', 'rb') as f:
    data = pickle.load(f)
print(data['image'].shape)  # Should be (H, W) or (3, H, W)
print(data['box'].shape)    # Should be (N, 4)
print(data['label'].shape)  # Should be (N,)
```

---

## 📚 Documentation

- **[CONFIG_GUIDE.md](CONFIG_GUIDE.md)**: Complete configuration reference
- **[WORKFLOW.md](WORKFLOW.md)**: Detailed step-by-step workflow
- **[Architecture.md](Architecture.md)**: Model architecture details

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Suraj Kumar**  
GitHub: [@surajcodesml](https://github.com/surajcodesml)  
Repository: [RCNN-OCT](https://github.com/surajcodesml/RCNN-OCT)

---

## 🙏 Acknowledgments

- PyTorch and torchvision teams for the excellent deep learning framework
- Faster R-CNN implementation by Ross Girshick et al.
- OCT imaging community for driving innovation in retinal diagnostics

---

## 📊 Citation

If you use this code in your research, please cite:

```bibtex
@software{rcnn_oct_2025,
  author = {Kumar, Suraj},
  title = {RCNN-OCT: Retinal OCT Object Detection Pipeline},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/surajcodesml/RCNN-OCT}
}
```

---

**Last Updated**: December 2025  
**Version**: 1.0.0
