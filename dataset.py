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

    image = np.asarray(image, dtype=np.float32)

    # Boxes retrieval
    if "boxes" in sample:
        boxes = sample["boxes"]
    elif "bboxes" in sample:
        boxes = sample["bboxes"]
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)

    boxes = np.asarray(boxes, dtype=np.float32)

    # Labels retrieval
    if "labels" in sample:
        labels = sample["labels"]
    else:
        labels = np.zeros((boxes.shape[0],), dtype=np.int64)

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

        # Filter out ignored labels
        valid_mask = labels_np != 2
        boxes_np = boxes_np[valid_mask]
        labels_np = labels_np[valid_mask]

        # Remap labels to contiguous values starting at 1
        if self.label_mapping:
            labels_np = np.array([self.label_mapping[int(lbl)] for lbl in labels_np], dtype=np.int64)
        else:
            labels_np = np.ones_like(labels_np, dtype=np.int64)

        # Prepare image tensor
        image_tensor = torch.from_numpy(image_np).float()
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 3 and image_tensor.shape[0] != 1:
            # Assume HWC
            image_tensor = image_tensor.permute(2, 0, 1)
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)
        image_tensor = image_tensor / 255.0

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
