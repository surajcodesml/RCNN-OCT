#!/bin/bash
# Complete workflow example for RCNN-OCT training and evaluation

set -e  # Exit on error

# Configuration file
CONFIG_FILE="config.yaml"

echo "=========================================="
echo "RCNN-OCT Complete Workflow"
echo "=========================================="
echo "Using configuration: $CONFIG_FILE"

# Step 1: Generate data splits
echo -e "\n[Step 1] Generating data splits..."
python split.py --config "$CONFIG_FILE"

echo -e "\n✓ Splits generated"

# Step 2: Train model
echo -e "\n[Step 2] Training model..."
python train.py --config "$CONFIG_FILE"

echo -e "\n✓ Training complete"

# Step 3: Run batch inference on test set
echo -e "\n[Step 3] Running inference on test set..."
python batch_inference.py \
  --config "$CONFIG_FILE" \
  --checkpoint checkpoints/best_model.pth

echo -e "\n✓ Inference complete"

# Summary
echo -e "\n=========================================="
echo "Workflow Complete!"
echo "=========================================="
echo "Check the following directories for results:"
echo "  - Splits: splits.json"
echo "  - Model: checkpoints/best_model.pth"
echo "  - Training logs: checkpoints/training_results.json"
echo "  - Test metrics: inference/test_metrics_*.json"
echo "  - Visualizations: inference/visualizations/"
echo "=========================================="
