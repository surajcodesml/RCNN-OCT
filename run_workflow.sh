#!/bin/bash
# Complete workflow example for RCNN-OCT training and evaluation

set -e  # Exit on error

# Configuration
DATA_ROOT="/home/suraj/Data/Nemours/pickle"
SPLITS_FILE="splits.json"
OUTPUT_DIR="checkpoints_test"
INFERENCE_DIR="inference"
EPOCHS=10
BATCH_SIZE=4
SEED=42

echo "=========================================="
echo "RCNN-OCT Complete Workflow"
echo "=========================================="

# Step 1: Generate data splits
echo -e "\n[Step 1] Generating data splits..."
python split.py \
  --data-root "$DATA_ROOT" \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed $SEED \
  --output "$SPLITS_FILE"

echo -e "\n✓ Splits generated: $SPLITS_FILE"

# Step 2: Train model
echo -e "\n[Step 2] Training model..."
python train.py \
  --splits-file "$SPLITS_FILE" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --output-dir "$OUTPUT_DIR" \
  --filter-empty 0.0 \
  --score-threshold 0.05 \
  --seed $SEED

echo -e "\n✓ Training complete: $OUTPUT_DIR/best_model.pth"

# Step 3: Run batch inference on test set
echo -e "\n[Step 3] Running inference on test set..."
python batch_inference.py \
  --checkpoint "$OUTPUT_DIR/best_model.pth" \
  --splits-file "$SPLITS_FILE" \
  --output-dir "$INFERENCE_DIR" \
  --score-threshold 0.5 \
  --visualize-samples 20

echo -e "\n✓ Inference complete: $INFERENCE_DIR/"

# Summary
echo -e "\n=========================================="
echo "Workflow Complete!"
echo "=========================================="
echo "Results:"
echo "  - Splits: $SPLITS_FILE"
echo "  - Model: $OUTPUT_DIR/best_model.pth"
echo "  - Training logs: $OUTPUT_DIR/training_results.json"
echo "  - Test metrics: $INFERENCE_DIR/test_metrics_*.json"
echo "  - Visualizations: $INFERENCE_DIR/visualizations/"
echo "=========================================="
