# RCNN-OCT

Minimal Faster R-CNN training and inference pipeline for retinal OCT B-scan object detection using PyTorch/torchvision.

## Repository structure
- `dataset.py`: Dataset loader for `.pkl` B-scan files, deterministic train/val split, and detection collate function.
- `model.py`: Faster R-CNN builder with a replaceable classification head sized to the dataset.
- `train.py`: Training loop with IoU-based validation metrics, learning-rate scheduling, and checkpointing.
- `inference.py`: Single-scan inference utilities plus matplotlib visualization.

## Data format
- Each `.pkl` file represents one 2D OCT B-scan and includes at least an `image` (or `img`) array, `boxes`, and `labels`.
- Filenames follow `patientID_LorR_instance_instanceNumber_bscanNumber.pkl` (e.g., `6_R_1_1001.pkl`). Each file is treated as an independent example even though volumes contain 31 B-scans.
- Images are stored as tensors with shape `(3, H, W)` in the range [0, 1] (typically 256×576 pixels).
- Bounding boxes are in **YOLO format** `[x_center, y_center, width, height]` with normalized coordinates [0, 1].
  - The dataset automatically converts these to corner format `[x1, y1, x2, y2]` in pixel coordinates for Faster R-CNN.
- Label handling:
  - **Original labels**: 0=Fovea, 1=SCR, 2=No boxes (ignored)
  - **Remapped for model**: Background=0 (automatic), Fovea=1, SCR=2
  - Label 2 ("no boxes") is filtered out during data loading
  - The model predicts 3 classes total: Background (0), Fovea (1), and SCR (2)

## Environment
- Python 3.10+, PyTorch with ROCm support, torchvision, numpy, matplotlib.
- Device selection uses `torch.device("cuda" if torch.cuda.is_available() else "cpu")` and will run on AMD GPUs with ROCm if available.

## Training
1. Place all `.pkl` files under a directory (files can be nested; they are discovered recursively).
2. Run training (example):
   ```bash
   python train.py --data-root /path/to/pkl/files --epochs 10 --batch-size 2 --val-ratio 0.2 --output-dir checkpoints
   ```
3. Printed metrics per epoch include average losses (`loss`, `loss_classifier`, `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`) and validation precision/recall/F1 at IoU ≥ 0.5.
4. The best model by validation F1 is saved to `<output-dir>/best_model.pth` together with the label mapping used for the run.

## Inference and visualization
Run inference on a single scan using a trained checkpoint:
```bash
python inference.py --checkpoint checkpoints/best_model.pth --sample /path/to/sample.pkl --score-threshold 0.5 --save-path pred.png
```
- Outputs filtered boxes, scores, and labels; if `--save-path` is provided, a visualization is written with predicted boxes overlaid.
- The checkpoint loader restores the saved label mapping so class indices stay consistent with training.

## Notes
- The dataset loader normalizes images to `[0, 1]` float32, converts grayscale to three channels, and asserts correct shapes/dtypes for detection models.
- Train/validation splits are deterministic via a fixed RNG seed; adjust `--seed` or `--val-ratio` if needed.
