# RCNN-OCT Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [Data Pipeline](#data-pipeline)
3. [Model Architecture](#model-architecture)
4. [Training Process](#training-process)
5. [Loss Functions](#loss-functions)
6. [Optimization](#optimization)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Inference](#inference)

---

## Overview

This project implements a Faster R-CNN object detection pipeline for retinal OCT (Optical Coherence Tomography) B-scan analysis. The system detects and localizes pathological features in 2D OCT scans using a state-of-the-art two-stage object detection architecture.

**Key Features:**
- Pre-trained Faster R-CNN with ResNet-50 backbone and Feature Pyramid Network (FPN)
- Custom classification head adapted to dataset-specific number of classes
- IoU-based evaluation metrics (Precision, Recall, F1-score)
- Automatic label remapping for handling discontinuous class indices
- Support for AMD GPUs via ROCm

---

## Data Pipeline

### Data Format

**Input Files:** `.pkl` (pickle) files containing OCT B-scans

Each pickle file contains:
```python
{
    "image" or "img": np.ndarray,  # 2D grayscale image or 3D RGB
    "box": np.ndarray,             # Bounding boxes in [cx, cy, w, h] normalized format
    "label": np.ndarray            # Class labels for each box
}
```

**File Naming Convention:**
```
patientID_LorR_instance_instanceNumber_bscanNumber.pkl
Example: 6_R_1_1001.pkl
```

### Data Loading Process (`dataset.py`)

#### 1. File Discovery
```python
find_pkl_files(root) → List[Path]
```
- Recursively searches directory tree for all `.pkl` files
- Returns sorted list for deterministic ordering

#### 2. Label Mapping
```python
build_label_mapping(files, ignore_label=2) → Dict[int, int]
```
**Purpose:** Create contiguous label indices starting from 1 (background is 0, handled internally by Faster R-CNN)

**Process:**
1. Scan all files to collect unique label values
2. Filter out `ignore_label=2` (typically "no pathology" cases)
3. Map remaining labels to contiguous integers: `{original_label: new_label}`

**Example:**
```python
Original labels: [0, 1, 3, 5]
After filtering label 2 and 0:
label_mapping = {1: 1, 3: 2, 5: 3}
num_classes = 4  # background (0) + 3 mapped classes
```

#### 3. Train/Validation Split
```python
split_files(files, val_ratio=0.2, seed=42)
```
- Deterministic random split using fixed seed
- Default: 80% training, 20% validation
- Maintains reproducibility across runs

#### 4. Image Preprocessing

**Bounding Box Conversion:**
Original format: `[cx, cy, w, h]` (normalized center coordinates + dimensions)
```python
# Convert to [x1, y1, x2, y2] pixel coordinates
x1 = (cx - w/2) * img_width
y1 = (cy - h/2) * img_height
x2 = (cx + w/2) * img_width
y2 = (cy + h/2) * img_height
```

**Filtering:**
- Remove boxes with all zeros (invalid/padding)
- Remove `label=2` (ignore class)
- Remove `label=0` (background/negative regions)

**Image Normalization:**
1. Convert to float32
2. Normalize pixel values to [0, 1] range
3. Convert grayscale (H, W) or (H, W, 1) to RGB (3, H, W)
4. Ensure CHW format (channels first) for PyTorch

**Target Dictionary:**
```python
target = {
    "boxes": Tensor[N, 4],      # Bounding boxes
    "labels": Tensor[N],         # Class labels (1 to num_classes-1)
    "image_id": Tensor[1],       # Unique identifier
    "area": Tensor[N],           # Box areas
    "iscrowd": Tensor[N]         # Always 0 (no crowd annotations)
}
```

#### 5. Data Loaders

**Training Loader:**
```python
DataLoader(
    batch_size=2,
    shuffle=True,
    num_workers=4,
    pin_memory=True,  # For GPU training
    collate_fn=detection_collate_fn
)
```

**Validation Loader:**
```python
DataLoader(
    batch_size=1,      # Single image for precise evaluation
    shuffle=False,
    num_workers=2,
    collate_fn=detection_collate_fn
)
```

**Custom Collate Function:**
```python
def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
```
- Returns lists instead of stacked tensors (images have varying sizes)
- Required for object detection where each image can have different numbers of objects

---

## Model Architecture

### Faster R-CNN Overview

Faster R-CNN is a two-stage object detection architecture:

**Stage 1: Region Proposal Network (RPN)**
- Generates candidate object proposals
- Predicts objectness scores (object vs. background)
- Regresses bounding box coordinates

**Stage 2: Detection Head**
- Classifies proposals into specific classes
- Refines bounding box coordinates

### Components

#### 1. Backbone: ResNet-50 with Feature Pyramid Network (FPN)

```python
fasterrcnn_resnet50_fpn(weights="DEFAULT")
```

**ResNet-50:**
- Deep convolutional neural network with 50 layers
- Uses residual connections to enable training very deep networks
- Pre-trained on ImageNet (1000 classes, millions of images)
- Extracts hierarchical features from input images

**Feature Pyramid Network (FPN):**
- Multi-scale feature extraction
- Combines low-resolution, semantically strong features with high-resolution, semantically weak features
- Creates a pyramid of feature maps at different scales
- Enables detection of objects at multiple sizes

**Feature Extraction Levels:**
```
Input Image (3, H, W)
    ↓
Conv1 → Conv2 → Conv3 → Conv4 → Conv5 (ResNet stages)
    ↓       ↓       ↓       ↓       ↓
  P2 ←─── P3 ←─── P4 ←─── P5    (FPN lateral connections)
  ↓       ↓       ↓       ↓
Multi-scale feature maps for RPN and ROI pooling
```

#### 2. Region Proposal Network (RPN)

**Purpose:** Generate object proposals (potential bounding boxes)

**Process:**
1. Slides a small network over FPN feature maps
2. At each location, predicts:
   - **Objectness scores:** Binary classification (object vs. background)
   - **Box deltas:** 4 values to refine anchor boxes

**Anchor Boxes:**
- Pre-defined boxes of various sizes and aspect ratios
- Typically 3 scales × 3 aspect ratios = 9 anchors per location
- Cover a wide range of object sizes and shapes

**Output:** 
- ~2000 region proposals per image
- Filtered by Non-Maximum Suppression (NMS) to remove overlapping proposals

#### 3. ROI (Region of Interest) Pooling

**Purpose:** Extract fixed-size features from variable-size proposals

**Process:**
1. Takes RPN proposals and FPN feature maps
2. Extracts features for each proposal region
3. Applies ROI Align (improved version of ROI Pooling)
4. Outputs fixed-size feature vectors (e.g., 7×7×256)

#### 4. Detection Head (Custom Classification Head)

```python
FastRCNNPredictor(in_features, num_classes)
```

**Default Head (replaced):**
- Designed for 91 COCO classes
- Not suitable for OCT-specific pathology detection

**Custom Head:**
```python
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

**Architecture:**
```
ROI Features (in_features-dimensional)
    ↓
Fully Connected Layer 1 (FC1)
    ↓
ReLU Activation
    ↓
Fully Connected Layer 2 (FC2)
    ↓
Output:
  - Class scores (num_classes)
  - Box regression (num_classes × 4)
```

**Outputs:**
- **Class scores:** Probability distribution over classes (including background)
- **Box deltas:** Refinement offsets for each class

**Number of Classes:**
```python
num_classes = len(label_mapping) + 1
# +1 for background class (implicitly handled by model)
```

### Complete Forward Pass

```
Input Image (3, H, W)
    ↓
ResNet-50 Backbone + FPN
    ↓
Multi-scale Feature Maps
    ↓
RPN (Region Proposal Network)
    ├─→ Objectness Scores
    └─→ ~2000 Proposals
    ↓
ROI Align (extract features for each proposal)
    ↓
Detection Head
    ├─→ Class Scores (num_classes)
    └─→ Box Refinements (num_classes × 4)
    ↓
Post-processing (NMS, score threshold)
    ↓
Final Detections: {boxes, scores, labels}
```

---

## Training Process

### Training Loop (`train.py`)

#### Hyperparameters

```python
{
    "epochs": 10,
    "batch_size": 2,
    "learning_rate": 0.005,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "lr_scheduler_step_size": 5,
    "lr_scheduler_gamma": 0.1,
    "val_ratio": 0.2,
    "score_threshold": 0.5,
    "patience": 7  # Early stopping
}
```

#### Single Epoch Training

```python
def train_one_epoch(model, optimizer, dataloader, device, epoch):
    model.train()
    for images, targets in dataloader:
        # Move data to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass (returns loss dictionary)
        loss_dict = model(images, targets)
        
        # Compute total loss
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
```

**Key Points:**
- Model in training mode automatically returns losses
- Each batch processes multiple images with varying numbers of objects
- Losses are summed and backpropagated together

#### Learning Rate Schedule

```python
StepLR(optimizer, step_size=5, gamma=0.1)
```

**Schedule:**
```
Epochs 1-5:   LR = 0.005
Epochs 6-10:  LR = 0.0005  (×0.1)
Epochs 11+:   LR = 0.00005 (×0.1)
```

**Purpose:** Gradually reduce learning rate for fine-tuning

#### Early Stopping

```python
patience = 7
if no improvement for 7 consecutive epochs:
    stop training
```

**Criteria:** Validation F1-score
- Prevents overfitting
- Saves computational resources
- Returns best model (by F1) instead of last epoch

#### Checkpointing

```python
torch.save({
    "model_state": model.state_dict(),
    "label_mapping": label_mapping
}, "best_model.pth")
```

**Saved Information:**
- Model weights (state_dict)
- Label mapping for inference consistency

**Trigger:** New best validation F1-score

---

## Loss Functions

Faster R-CNN optimizes 4 loss components simultaneously:

### 1. RPN Classification Loss (`loss_objectness`)

**Purpose:** Train RPN to distinguish objects from background

**Type:** Binary Cross-Entropy Loss

**Computation:**
```python
L_obj = BCE(predicted_objectness, ground_truth_objectness)
```

**Ground Truth:**
- Positive samples: Anchors with IoU > 0.7 with any ground truth box
- Negative samples: Anchors with IoU < 0.3 with all ground truth boxes
- Ignored: Anchors with 0.3 ≤ IoU ≤ 0.7

### 2. RPN Box Regression Loss (`loss_rpn_box_reg`)

**Purpose:** Train RPN to refine anchor boxes

**Type:** Smooth L1 Loss (Huber Loss)

**Computation:**
```python
L_rpn_box = SmoothL1(predicted_deltas, target_deltas)
```

**Only computed for positive anchors**

**Box Parameterization:**
```python
# Transform box coordinates to deltas
tx = (x - xa) / wa
ty = (y - ya) / ha
tw = log(w / wa)
th = log(h / ha)
```
Where `(x, y, w, h)` are target box parameters and `(xa, ya, wa, ha)` are anchor parameters.

### 3. Classification Loss (`loss_classifier`)

**Purpose:** Classify ROI proposals into specific classes

**Type:** Cross-Entropy Loss

**Computation:**
```python
L_cls = CrossEntropy(predicted_class_scores, ground_truth_labels)
```

**Number of Classes:** `num_classes` (including background)

**Sampling:**
- Positive samples: Proposals with IoU > 0.5 with ground truth
- Negative samples: Proposals with IoU < 0.5
- Balanced sampling: 25% positive, 75% negative (configurable)

### 4. Box Regression Loss (`loss_box_reg`)

**Purpose:** Refine bounding box coordinates for detected objects

**Type:** Smooth L1 Loss

**Computation:**
```python
L_box = SmoothL1(predicted_box_deltas, target_box_deltas)
```

**Only computed for positive ROIs**

### Total Loss

```python
L_total = L_obj + L_rpn_box + λ_cls * L_cls + λ_box * L_box
```

**Default weights:** All λ = 1.0 (equal weighting)

**Typical Loss Values (during training):**
```
Epoch 1:
  loss_objectness:   0.3-0.5
  loss_rpn_box_reg:  0.2-0.4
  loss_classifier:   0.5-1.0
  loss_box_reg:      0.3-0.6
  Total loss:        1.5-2.5

Converged (Epoch 10):
  loss_objectness:   0.05-0.1
  loss_rpn_box_reg:  0.05-0.1
  loss_classifier:   0.1-0.3
  loss_box_reg:      0.1-0.2
  Total loss:        0.3-0.7
```

---

## Optimization

### Optimizer: Stochastic Gradient Descent (SGD)

```python
SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)
```

#### Parameters

**Learning Rate (lr=0.005):**
- Controls step size in gradient descent
- Higher values: faster learning but less stable
- Lower values: slower but more stable convergence

**Momentum (0.9):**
- Accelerates SGD in relevant directions
- Dampens oscillations
- Helps escape local minima

**Weight Decay (5e-4):**
- L2 regularization penalty
- Prevents overfitting by penalizing large weights
- Formula: `loss += 0.0005 * sum(w²)` for all weights

#### Why SGD over Adam?

**Advantages:**
- Better generalization for computer vision tasks
- More stable training for object detection
- Works well with pre-trained weights
- Standard choice for Faster R-CNN

### Gradient Computation

```python
# Forward pass
loss_dict = model(images, targets)
total_loss = sum(loss_dict.values())

# Backward pass
optimizer.zero_grad()      # Clear previous gradients
total_loss.backward()       # Compute gradients
optimizer.step()            # Update weights
```

**Automatic Differentiation:**
- PyTorch autograd tracks all operations
- Computes gradients through entire network
- Updates all trainable parameters

### Learning Rate Scheduling

```python
StepLR(optimizer, step_size=5, gamma=0.1)
```

**Effect:**
```python
for epoch in range(1, 11):
    train_one_epoch(...)
    scheduler.step()
    # Epoch 1-5:  lr = 0.005
    # Epoch 6-10: lr = 0.0005
    # Epoch 11+:  lr = 0.00005
```

**Benefits:**
- Initial epochs: large steps for coarse optimization
- Later epochs: small steps for fine-tuning
- Prevents overshooting optimal solution

---

## Evaluation Metrics

### Intersection over Union (IoU)

**Definition:** Overlap between predicted and ground truth boxes

```python
def compute_iou(box_a, box_b):
    # Intersection
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    
    return intersection / union
```

**IoU Threshold:** 0.5 (standard for object detection)
- IoU ≥ 0.5: True Positive
- IoU < 0.5: False Positive

### Detection Matching Algorithm

```python
for each predicted_box:
    compute IoU with all ground_truth_boxes
    best_match = ground_truth_box with highest IoU
    
    if IoU(predicted_box, best_match) >= 0.5 and best_match not yet matched:
        True Positive (TP)
        mark best_match as matched
    else:
        False Positive (FP)

# Unmatched ground truth boxes
False Negatives (FN) = total_ground_truth - matched_ground_truth
```

### Core Metrics

#### 1. Precision

**Formula:**
```
Precision = TP / (TP + FP)
```

**Interpretation:**
- "Of all detections made, what fraction were correct?"
- High precision: few false alarms
- Range: [0, 1], higher is better

**Example:**
```
TP = 80, FP = 20
Precision = 80 / (80 + 20) = 0.80 (80%)
```

#### 2. Recall (Sensitivity)

**Formula:**
```
Recall = TP / (TP + FN)
```

**Interpretation:**
- "Of all ground truth objects, what fraction were detected?"
- High recall: few missed detections
- Range: [0, 1], higher is better

**Example:**
```
TP = 80, FN = 10
Recall = 80 / (80 + 10) = 0.889 (88.9%)
```

#### 3. F1-Score

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretation:**
- Harmonic mean of precision and recall
- Balances both metrics
- Used for model selection (best checkpoint)
- Range: [0, 1], higher is better

**Example:**
```
Precision = 0.80, Recall = 0.889
F1 = 2 × (0.80 × 0.889) / (0.80 + 0.889) = 0.842
```

### Score Threshold

**Purpose:** Filter low-confidence predictions

```python
keep = predicted_scores >= score_threshold  # default: 0.5
filtered_boxes = predicted_boxes[keep]
filtered_scores = predicted_scores[keep]
```

**Effect:**
- Higher threshold: fewer predictions, higher precision, lower recall
- Lower threshold: more predictions, lower precision, higher recall

**Typical Output:**
```
Epoch 10/10: loss=0.45, cls=0.15, box=0.12, obj=0.08, rpn_box=0.10 | 
P=0.85, R=0.82, F1=0.84 (TP=120, FP=21, FN=26)
```

---

## Inference

### Inference Process (`inference.py`)

#### 1. Load Checkpoint

```python
checkpoint = torch.load("best_model.pth")
model.load_state_dict(checkpoint["model_state"])
label_mapping = checkpoint["label_mapping"]
```

**Restores:**
- Trained model weights
- Label mapping (ensures consistent class interpretation)

#### 2. Prepare Input

```python
# Load sample
sample = pickle.load(open("sample.pkl", "rb"))
image, boxes, labels = extract_arrays(sample)

# Preprocess (same as training)
image_tensor = preprocess(image)  # Normalize, convert to RGB, CHW format
```

#### 3. Forward Pass (Inference Mode)

```python
model.eval()  # Disable dropout, use running stats for batch norm
with torch.no_grad():  # Disable gradient computation
    predictions = model([image_tensor])
```

**Model Output:**
```python
predictions = [{
    "boxes": Tensor[N, 4],      # Predicted bounding boxes
    "scores": Tensor[N],         # Confidence scores [0, 1]
    "labels": Tensor[N]          # Predicted class labels
}]
```

#### 4. Post-processing

**Score Filtering:**
```python
keep = predictions["scores"] >= score_threshold
filtered_boxes = predictions["boxes"][keep]
filtered_scores = predictions["scores"][keep]
filtered_labels = predictions["labels"][keep]
```

**Non-Maximum Suppression (NMS):**
- Already applied internally by Faster R-CNN
- Removes duplicate detections for the same object
- Keeps highest-scoring box among overlapping detections

#### 5. Visualization

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1)
ax.imshow(image, cmap='gray')

for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    
    rect = patches.Rectangle(
        (x1, y1), width, height,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    
    # Add label text
    ax.text(x1, y1 - 5, f"Class {label}: {score:.2f}",
            bbox=dict(facecolor='red', alpha=0.5),
            fontsize=10, color='white')

plt.savefig("prediction.png")
```

### Inference Example

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --sample data/6_R_1_1001.pkl \
    --score-threshold 0.7 \
    --save-path output/pred.png
```

**Output:**
```
Loaded checkpoint: checkpoints/best_model.pth
Label mapping: {1: 1, 3: 2, 5: 3}
Detections: 3
  Box 1: [120, 45, 180, 95], Score: 0.92, Label: 1
  Box 2: [250, 60, 310, 110], Score: 0.85, Label: 2
  Box 3: [400, 75, 450, 125], Score: 0.78, Label: 1
Visualization saved to: output/pred.png
```

---

## Training Results and Monitoring

### Results Storage (`training_results.json`)

Saved after training completion:

```json
{
  "hyperparameters": {
    "epochs": 10,
    "batch_size": 2,
    "learning_rate": 0.005,
    "num_classes": 4,
    "train_samples": 800,
    "val_samples": 200
  },
  "training_history": [
    {
      "epoch": 1,
      "train_loss": 1.85,
      "val_precision": 0.45,
      "val_recall": 0.52,
      "val_f1": 0.48
    },
    ...
  ],
  "best_metrics": {
    "best_f1": 0.84,
    "best_epoch": 8
  }
}
```

### Monitoring Training Progress

**Console Output:**
```
Epoch 1/10: 100%|████████| 400/400 [05:23<00:00]
loss=1.85, cls=0.65, box=0.45, obj=0.42, rpn_box=0.33 | 
P=0.45, R=0.52, F1=0.48 (TP=85, FP=104, FN=78)

Epoch 2/10: 100%|████████| 400/400 [05:21<00:00]
loss=1.12, cls=0.38, box=0.28, obj=0.25, rpn_box=0.21 | 
P=0.68, R=0.71, F1=0.69 (TP=116, FP=55, FN=47)
Saved new best model to checkpoints/best_model.pth with F1=0.69

...

Epoch 8/10: 100%|████████| 400/400 [05:19<00:00]
loss=0.45, cls=0.15, box=0.12, obj=0.08, rpn_box=0.10 | 
P=0.85, R=0.82, F1=0.84 (TP=134, FP=24, FN=29)
Saved new best model to checkpoints/best_model.pth with F1=0.84

Epoch 9/10: No improvement for 1 epoch(s)
Epoch 10/10: No improvement for 2 epoch(s)

Training completed. Best F1: 0.84 at epoch 8
```

---

## Summary

This Faster R-CNN implementation for OCT B-scan detection provides:

1. **Robust Data Pipeline:** Handles varying label formats, automatic label remapping, deterministic splits
2. **State-of-the-Art Architecture:** Pre-trained ResNet-50 + FPN backbone with custom detection head
3. **Comprehensive Training:** Multi-component loss optimization, learning rate scheduling, early stopping
4. **Rigorous Evaluation:** IoU-based precision, recall, F1-score at 0.5 threshold
5. **Production-Ready Inference:** Checkpoint loading, score filtering, visualization utilities

**Key Design Choices:**
- **Pre-trained backbone:** Leverages ImageNet knowledge for better feature extraction
- **SGD optimizer:** Standard for object detection, better generalization than Adam
- **F1-score selection:** Balances precision and recall for model selection
- **Early stopping:** Prevents overfitting, saves computation
- **Deterministic splits:** Ensures reproducible results

This architecture is well-suited for medical imaging applications requiring precise localization of pathological features in OCT scans