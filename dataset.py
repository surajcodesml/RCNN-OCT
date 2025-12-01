"""Dataset and data loading utilities for OCT object detection."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def find_pkl_files(root: Path) -> List[Path]:
    """Recursively collect all `.pkl` files under ``root``.

    Args:
        root: Directory containing OCT scan pickle files.

    Returns:
        Sorted list of paths to pickle files.
    """
    root = root.expanduser().resolve()
    return sorted(root.rglob("*.pkl"))


def _load_pickle(path: Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _extract_arrays(sample: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract image, boxes, and labels arrays from a loaded pickle sample."""
    if not isinstance(sample, dict):
        raise TypeError(f"Expected sample dict, got {type(sample)}")

    # Image retrieval
    if "image" in sample:
        image = sample["image"]
    elif "img" in sample:
        image = sample["img"]
    else:
        raise KeyError("Sample must contain 'image' or 'img' key")

    # Convert torch tensors to numpy if needed
    if hasattr(image, 'numpy'):
        image = image.numpy()
    image = np.asarray(image, dtype=np.float32)

    # Boxes retrieval - NOTE: singular 'box' is the correct key name!
    if "boxes" in sample:
        boxes = sample["boxes"]
    elif "bboxes" in sample:
        boxes = sample["bboxes"]
    elif "box" in sample:  # DEBUG: Added support for singular 'box' key
        boxes = sample["box"]
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)

    # Convert torch tensors to numpy if needed
    if hasattr(boxes, 'numpy'):
        boxes = boxes.numpy()
    boxes = np.asarray(boxes, dtype=np.float32)

    # Labels retrieval - NOTE: singular 'label' is the correct key name!
    if "labels" in sample:
        labels = sample["labels"]
    elif "label" in sample:  # DEBUG: Added support for singular 'label' key
        labels = sample["label"]
    else:
        labels = np.zeros((boxes.shape[0],), dtype=np.int64)

    # Convert torch tensors to numpy if needed
    if hasattr(labels, 'numpy'):
        labels = labels.numpy()
    labels = np.asarray(labels, dtype=np.int64)
    return image, boxes, labels


def build_label_mapping(files: Sequence[Path], ignore_label: int | None = 2) -> Dict[int, int]:
    """Build a contiguous label mapping from dataset files.

    Args:
        files: Iterable of pickle file paths.
        ignore_label: Label value to skip when computing classes (default: 2 for "no boxes").

    Returns:
        Mapping from original labels to contiguous values starting at 1.
        Example: {0: 1, 1: 2} means Fovea→class1, SCR→class2
    """
    label_set: set[int] = set()
    for path in files:
        sample = _load_pickle(path)
        _, _, labels = _extract_arrays(sample)
        if labels.size == 0:
            continue
        for value in labels.tolist():
            if ignore_label is not None and value == ignore_label:
                continue
            label_set.add(int(value))
    sorted_labels = sorted(label_set)
    return {label: idx + 1 for idx, label in enumerate(sorted_labels)} if sorted_labels else {}


def split_files(files: Sequence[Path], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Path], List[Path]]:
    """Deterministically split files into train and validation subsets."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(files))
    rng.shuffle(indices)
    split = int(len(files) * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    train_files = [files[i] for i in sorted(train_idx.tolist())]
    val_files = [files[i] for i in sorted(val_idx.tolist())]
    return train_files, val_files


def detection_collate_fn(batch: Iterable[Tuple[torch.Tensor, dict]]) -> Tuple[List[torch.Tensor], List[dict]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


class OCTDetectionDataset(Dataset):
    """Dataset for OCT B-scan object detection from pickle files."""

    def __init__(
        self,
        files: Sequence[Path],
        label_mapping: Dict[int, int] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        filter_empty_ratio: float = 0.0,
        seed: int = 42,
    ) -> None:
        """
        Args:
            files: List of pickle file paths
            label_mapping: Mapping from original labels to contiguous class IDs
            transform: Optional transform to apply to images
            filter_empty_ratio: Fraction of empty images (no boxes) to remove (0.0-1.0).
                               0.0 = keep all images, 0.5 = remove 50% of empty images,
                               1.0 = remove all empty images
            seed: Random seed for reproducible filtering
        """
        self.transform = transform
        self.label_mapping = label_mapping if label_mapping is not None else build_label_mapping(files)
        self.num_classes = len(self.label_mapping) + 1 if self.label_mapping else 2
        
        # Filter empty images if requested
        if filter_empty_ratio > 0.0:
            self.files = self._filter_empty_images(files, filter_empty_ratio, seed)
        else:
            self.files = list(files)

    def _filter_empty_images(self, files: Sequence[Path], filter_ratio: float, seed: int) -> List[Path]:
        """Filter out a percentage of images without bounding boxes.
        
        Args:
            files: All pickle file paths
            filter_ratio: Fraction of empty images to remove (0.0-1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Filtered list of file paths with some empty images removed
        """
        rng = np.random.default_rng(seed)
        files_with_boxes = []
        files_without_boxes = []
        
        print(f"Filtering empty images (removing {filter_ratio*100:.0f}% of images without boxes)...")
        
        for path in files:
            sample = _load_pickle(path)
            _, boxes_np, labels_np = _extract_arrays(sample)
            
            # Check if file has any valid boxes (non-zero boxes with valid labels)
            # Label 0 = Fovea (valid), Label 1 = SCR (valid), Label 2 = No boxes (invalid)
            valid_mask = np.ones(len(boxes_np), dtype=bool)
            zero_boxes = np.all(boxes_np == 0, axis=1) if boxes_np.size > 0 else np.array([], dtype=bool)
            valid_mask &= ~zero_boxes if zero_boxes.size > 0 else True
            # Only filter out label=2 (no boxes)
            valid_mask &= (labels_np != 2) if labels_np.size > 0 else True
            
            has_boxes = np.any(valid_mask)
            
            if has_boxes:
                files_with_boxes.append(path)
            else:
                files_without_boxes.append(path)
        
        # Keep all files with boxes
        kept_files = files_with_boxes.copy()
        
        # Keep only (1 - filter_ratio) of files without boxes
        num_empty_to_keep = int(len(files_without_boxes) * (1.0 - filter_ratio))
        if num_empty_to_keep > 0:
            # Randomly select which empty files to keep
            indices = rng.choice(len(files_without_boxes), size=num_empty_to_keep, replace=False)
            kept_empty = [files_without_boxes[i] for i in sorted(indices)]
            kept_files.extend(kept_empty)
        
        print(f"  Files with boxes: {len(files_with_boxes)} (kept all)")
        print(f"  Files without boxes: {len(files_without_boxes)} (kept {num_empty_to_keep}, removed {len(files_without_boxes) - num_empty_to_keep})")
        print(f"  Total files: {len(files)} → {len(kept_files)}")
        
        return sorted(kept_files)
    
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        path = self.files[idx]
        sample = _load_pickle(path)
        image_np, boxes_np, labels_np = _extract_arrays(sample)

        if labels_np.shape[0] != boxes_np.shape[0]:
            raise ValueError(f"Mismatched labels and boxes for {path}")

        # Filter out boxes with all zeros (invalid/padding boxes)
        # Only filter label=2 which means "no bounding boxes"
        # Label 0 = Fovea (KEEP), Label 1 = SCR (KEEP), Label 2 = No boxes (REMOVE)
        valid_mask = np.ones(len(boxes_np), dtype=bool)
        
        # Remove boxes that are all zeros (invalid/padding boxes)
        zero_boxes = np.all(boxes_np == 0, axis=1)
        valid_mask &= ~zero_boxes
        
        # Remove only label=2 (no bounding boxes / ignore)
        valid_mask &= (labels_np != 2)
        
        boxes_np = boxes_np[valid_mask]
        labels_np = labels_np[valid_mask]
        
        # Convert boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2]
        # The data format is: [cx, cy, w, h] where all values are normalized [0, 1]
        # Convert to pixel coordinates for Faster R-CNN
        if boxes_np.shape[0] > 0:
            # Get image dimensions - image_np is in CHW format (3, H, W)
            if image_np.ndim == 3:
                img_height, img_width = image_np.shape[1], image_np.shape[2]
            else:
                # Fallback for HW format
                img_height, img_width = image_np.shape[0], image_np.shape[1]
            
            cx, cy, w, h = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
            x1 = (cx - w / 2) * img_width
            y1 = (cy - h / 2) * img_height
            x2 = (cx + w / 2) * img_width
            y2 = (cy + h / 2) * img_height
            boxes_np = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
            
            # Clamp boxes to image boundaries
            boxes_np[:, 0] = np.clip(boxes_np[:, 0], 0, img_width)
            boxes_np[:, 1] = np.clip(boxes_np[:, 1], 0, img_height)
            boxes_np[:, 2] = np.clip(boxes_np[:, 2], 0, img_width)
            boxes_np[:, 3] = np.clip(boxes_np[:, 3], 0, img_height)

        # Remap labels to contiguous values starting at 1
        if self.label_mapping:
            labels_np = np.array([self.label_mapping[int(lbl)] for lbl in labels_np], dtype=np.int64)
        else:
            labels_np = np.ones_like(labels_np, dtype=np.int64)

        # Prepare image tensor
        image_tensor = torch.from_numpy(image_np).float()
        
        # Normalize to [0, 1] range
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        # Handle different image formats: HW, CHW, or HWC
        if image_tensor.ndim == 2:
            # Grayscale (H, W) -> add channel dimension
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 3:
            # Check if it's HWC (height, width, channels) format
            if image_tensor.shape[2] in [1, 3] and image_tensor.shape[0] > 3:
                # Likely HWC format, convert to CHW
                image_tensor = image_tensor.permute(2, 0, 1)
        
        # Convert to 3-channel RGB if needed
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)
        elif image_tensor.shape[0] != 3:
            raise ValueError(f"Unexpected number of channels: {image_tensor.shape[0]}")

        boxes_tensor = torch.from_numpy(boxes_np).float()
        labels_tensor = torch.from_numpy(labels_np).long()

        if boxes_tensor.numel() == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        # Target dictionary
        area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1]) if boxes_tensor.numel() > 0 else torch.zeros((0,), dtype=torch.float32)
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((labels_tensor.shape[0],), dtype=torch.int64),
        }

        self._assert_valid(image_tensor, target)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return image_tensor, target

    @staticmethod
    def _assert_valid(image: torch.Tensor, target: dict) -> None:
        assert image.dtype == torch.float32, "Image must be float32"
        assert image.dim() == 3 and image.shape[0] == 3, "Image must have shape (3, H, W)"
        boxes = target["boxes"]
        labels = target["labels"]
        assert boxes.dtype == torch.float32 and boxes.ndim == 2 and boxes.shape[1] == 4, "Boxes must be float32 with shape (N, 4)"
        assert labels.dtype == torch.int64, "Labels must be int64"
        assert target["image_id"].dtype == torch.int64 and target["image_id"].numel() == 1
        assert target["iscrowd"].dtype == torch.int64


def create_datasets(
    root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    filter_empty_ratio: float = 0.0,
    max_samples: int | None = None,
) -> Tuple[OCTDetectionDataset, OCTDetectionDataset, Dict[int, int]]:
    """Create train and validation datasets with a deterministic split.
    
    Args:
        root: Root directory containing pickle files
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducibility
        filter_empty_ratio: Fraction of empty images (no boxes) to remove from training set.
                          0.0 = keep all images (default), 0.5 = remove 50% of empty images
    
    Returns:
        Tuple of (train_dataset, val_dataset, label_mapping)
    """
    files = find_pkl_files(root)
    
    # Limit samples for testing if requested
    if max_samples is not None and max_samples > 0:
        files = files[:max_samples]
        print(f"Limited to {len(files)} samples for testing")
    
    label_mapping = build_label_mapping(files, ignore_label=2)  # Ignore label 2 (no boxes)
    train_files, val_files = split_files(files, val_ratio, seed)
    
    # Apply filtering only to training set, keep validation set intact for fair evaluation
    train_dataset = OCTDetectionDataset(
        train_files, 
        label_mapping, 
        filter_empty_ratio=filter_empty_ratio, 
        seed=seed)
    val_dataset = OCTDetectionDataset(val_files, label_mapping, filter_empty_ratio=0.0)  # Never filter validation
    
    return train_dataset, val_dataset, label_mapping
