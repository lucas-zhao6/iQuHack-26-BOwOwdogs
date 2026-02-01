"""Utilities for generating and loading fixed CV splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def split_train_valid_groups(
    indices: np.ndarray,
    groups: np.ndarray,
    valid_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-aware train/valid split for early stopping."""
    from sklearn.model_selection import GroupShuffleSplit

    unique_groups = np.unique(groups[indices])
    if len(unique_groups) < 3 or valid_size <= 0:
        return indices, np.array([], dtype=int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
    train_local, val_local = next(
        splitter.split(np.zeros(len(indices)), groups=groups[indices])
    )
    return indices[train_local], indices[val_local]


def get_valid_group_splits(
    X: np.ndarray,
    y_threshold: np.ndarray,
    groups: np.ndarray,
    splitter,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return group splits where training contains all observed labels."""
    from .scoring import threshold_to_idx

    required_labels = {threshold_to_idx(v) for v in y_threshold}
    valid_splits: list[tuple[np.ndarray, np.ndarray]] = []
    total_splits = 0

    for train_idx, test_idx in splitter.split(X, y_threshold, groups):
        total_splits += 1
        train_labels = {threshold_to_idx(v) for v in y_threshold[train_idx]}
        if required_labels.issubset(train_labels):
            valid_splits.append((train_idx, test_idx))

    if not valid_splits:
        raise ValueError(
            "No valid group splits contain all threshold labels. "
            "Reduce n_splits or adjust grouping."
        )

    if len(valid_splits) < total_splits:
        print(
            f"  Using {len(valid_splits)}/{total_splits} splits with full label coverage"
        )

    return valid_splits


def build_cv_splits(
    X: np.ndarray,
    y_threshold: np.ndarray,
    groups: np.ndarray,
    splitter,
    valid_size: float = 0.2,
    seed: int = 42,
) -> list[dict]:
    """Build CV splits with fixed train/val/test indices."""
    base_splits = get_valid_group_splits(X, y_threshold, groups, splitter)
    splits = []
    for fold, (train_idx, test_idx) in enumerate(base_splits):
        train_idx, val_idx = split_train_valid_groups(
            train_idx, groups, valid_size=valid_size, seed=seed + fold
        )
        splits.append(
            {
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            }
        )
    return splits


def save_cv_splits(
    path: str | Path,
    splits: list[dict],
    n_samples: int,
    metadata: dict | None = None,
) -> None:
    path = Path(path)
    payload = {
        "version": 1,
        "n_samples": int(n_samples),
        "metadata": metadata or {},
        "splits": [
            {
                "train_idx": s["train_idx"].tolist(),
                "val_idx": s["val_idx"].tolist(),
                "test_idx": s["test_idx"].tolist(),
            }
            for s in splits
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_cv_splits(path: str | Path) -> tuple[list[dict], dict]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    splits = []
    for s in data["splits"]:
        splits.append(
            {
                "train_idx": np.array(s["train_idx"], dtype=int),
                "val_idx": np.array(s["val_idx"], dtype=int),
                "test_idx": np.array(s["test_idx"], dtype=int),
            }
        )
    metadata = data.get("metadata", {})
    metadata["n_samples"] = data.get("n_samples")
    return splits, metadata
