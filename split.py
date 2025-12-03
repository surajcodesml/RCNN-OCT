"""Generate train/val/test splits for OCT detection dataset."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def create_splits(
    data_root: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_path: Path = Path("splits.json"),
) -> Dict[str, List[str]]:
    """
    Split pickle files into train/val/test sets.
    
    Args:
        data_root: Root directory containing .pkl files
        train_ratio: Fraction of data for training (default: 0.8)
        val_ratio: Fraction of data for validation (default: 0.1)
        test_ratio: Fraction of data for testing (default: 0.1)
        seed: Random seed for reproducibility
        output_path: Path to save splits.json file
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing file paths
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
        raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total_ratio}")
    
    # Find all pickle files
    data_root = data_root.expanduser().resolve()
    all_files = sorted(data_root.rglob("*.pkl"))
    
    if len(all_files) == 0:
        raise ValueError(f"No .pkl files found in {data_root}")
    
    print(f"Found {len(all_files)} pickle files in {data_root}")
    
    # Shuffle files with fixed seed
    random.seed(seed)
    random.shuffle(all_files)
    
    # Calculate split indices
    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # Remaining goes to test to ensure all files are used
    n_test = n_total - n_train - n_val
    
    # Split files
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]
    
    # Convert to strings (relative to data_root for portability)
    splits = {
        "train": [str(f) for f in train_files],
        "val": [str(f) for f in val_files],
        "test": [str(f) for f in test_files],
        "metadata": {
            "data_root": str(data_root),
            "train_count": len(train_files),
            "val_count": len(val_files),
            "test_count": len(test_files),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
        }
    }
    
    # Save to JSON
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSplit summary:")
    print(f"  Train: {len(train_files)} files ({len(train_files)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_files)} files ({len(val_files)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_files)} files ({len(test_files)/n_total*100:.1f}%)")
    print(f"\nSplits saved to: {output_path}")
    
    return splits


def load_splits(splits_path: Path) -> Dict[str, List[str]]:
    """Load splits from JSON file."""
    with open(splits_path, "r") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate train/val/test splits for OCT dataset")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/suraj/Data/Nemours/pickle"),
        help="Root directory containing .pkl files"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for testing (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("splits.json"),
        help="Output path for splits JSON file (default: splits.json)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    create_splits(
        data_root=args.data_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
