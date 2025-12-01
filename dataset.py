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
        ignore_label: Label value to skip when computing classes.

    Returns:
        Mapping from original labels to contiguous values starting at 1.
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
    ) -> None:
        self.files = list(files)
        self.transform = transform
        self.label_mapping = label_mapping if label_mapping is not None else build_label_mapping(self.files)
        self.num_classes = len(self.label_mapping) + 1 if self.label_mapping else 2

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        path = self.files[idx]
        sample = _load_pickle(path)
        image_np, boxes_np, labels_np = _extract_arrays(sample)

        if labels_np.shape[0] != boxes_np.shape[0]:
            raise ValueError(f"Mismatched labels and boxes for {path}")

        # Filter out boxes with all zeros (invalid/padding boxes)
        # Also filter label=2 (ignore label) and label=0 (background/negative)
        valid_mask = np.ones(len(boxes_np), dtype=bool)
        
        # Remove boxes that are all zeros
        zero_boxes = np.all(boxes_np == 0, axis=1)
        valid_mask &= ~zero_boxes
        
        # Remove ignore label (2) - this appears to be "no pathology" or similar
        valid_mask &= (labels_np != 2)
        
        # Remove background/negative label (0) - this appears to be non-target regions
        valid_mask &= (labels_np != 0)
        
        boxes_np = boxes_np[valid_mask]
        labels_np = labels_np[valid_mask]
        
        # Convert boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2]
        # The data format is: [cx, cy, w, h] where all values are normalized [0, 1]
        # Convert to pixel coordinates for Faster R-CNN
        if boxes_np.shape[0] > 0:
            # Get image dimensions (after conversion to CHW format)
            img_height, img_width = image_np.shape[1], image_np.shape[2] if image_np.ndim == 3 else (image_np.shape[0], image_np.shape[1])
            
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
) -> Tuple[OCTDetectionDataset, OCTDetectionDataset, Dict[int, int]]:
    """Create train and validation datasets with a deterministic split."""
    files = find_pkl_files(root)
    train_files, val_files = split_files(files, val_ratio=val_ratio, seed=seed)
    label_mapping = build_label_mapping(files)
    train_dataset = OCTDetectionDataset(train_files, label_mapping)
    val_dataset = OCTDetectionDataset(val_files, label_mapping)
    return train_dataset, val_dataset, label_mapping
