# Empty Image Filtering Feature

## Overview
The dataset contains many images without bounding boxes (imbalanced dataset). This feature allows you to reduce the number of empty images in the training set while keeping ALL images with annotations.

## How It Works

### Key Points:
- **Keeps ALL images with bounding boxes** (fovea/SCR annotations)
- **Removes a percentage of images WITHOUT bounding boxes**
- **Validation set is NEVER filtered** - always uses full validation data for fair evaluation
- **Reproducible** - uses the same seed for consistent filtering

### Parameters:
- `--filter-empty 0.0` : Keep all images (default, no filtering)
- `--filter-empty 0.5` : Remove 50% of empty images
- `--filter-empty 0.75`: Remove 75% of empty images  
- `--filter-empty 1.0` : Remove all empty images (only train on images with boxes)

## Usage Examples

### Full Dataset (No Filtering)
```bash
python train.py --batch-size 2 --epochs 10
# or explicitly:
python train.py --batch-size 2 --epochs 10 --filter-empty 0.0
```

### Reduce Empty Images by 50%
```bash
python train.py --batch-size 4 --epochs 10 --filter-empty 0.5
```
This allows you to use 2x larger batch size since you have ~50% fewer samples.

### Reduce Empty Images by 75%
```bash
python train.py --batch-size 8 --epochs 10 --filter-empty 0.75
```
Even larger batch sizes possible with fewer samples.

### Train Only on Images With Annotations
```bash
python train.py --batch-size 16 --epochs 10 --filter-empty 1.0
```

## Benefits

1. **Memory Efficiency**: Fewer samples = can use larger batch sizes
2. **Faster Training**: Less data to process per epoch
3. **Better Class Balance**: More balanced ratio of positive/negative examples
4. **Easy Toggle**: Just change one parameter to enable/disable

## Technical Details

### Implementation:
- Filtering happens in `OCTDetectionDataset.__init__()` via `_filter_empty_images()`
- Scans all files once to identify which have valid bounding boxes
- Uses numpy random generator with seed for reproducibility
- Label 0 and 2 are treated as "no annotation" (filtered out)
- Boxes with all zeros `[0, 0, 0, 0]` are treated as empty

### What Counts as "Empty":
An image is considered empty if it has:
- No boxes at all, OR
- Only boxes with all zeros `[0, 0, 0, 0]`, OR  
- Only labels of 0 (background) or 2 (ignore)

### What Gets Kept:
An image is kept if it has:
- At least one box with label=1 (valid annotation for fovea/SCR)
- Non-zero box coordinates

## Example Output

When you run with `--filter-empty 0.5`:
```
Filtering empty images (removing 50% of images without boxes)...
  Files with boxes: 1234 (kept all)
  Files without boxes: 8864 (kept 4432, removed 4432)
  Total files: 10098 → 5666
```

## Saved in Training Results

The `filter_empty_ratio` is saved in `training_results.json` so you know what filtering was used:
```json
{
  "hyperparameters": {
    "filter_empty_ratio": 0.5,
    ...
  }
}
```

## Recommendations

- Start with `--filter-empty 0.5` to reduce dataset size while maintaining some negatives
- Use `--filter-empty 0.75` if you have memory constraints
- Keep `--filter-empty 0.0` (default) for final training to use all available data
- The validation set is never filtered to ensure fair evaluation
