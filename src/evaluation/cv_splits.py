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


def get_balanced_group_splits(
    X: np.ndarray,
    y_threshold: np.ndarray,
    groups: np.ndarray,
    splitter,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return group splits with label coverage on both sides when possible."""
    from .scoring import threshold_to_idx

    # Map group -> set of labels present in that group
    group_labels: dict[str, set[int]] = {}
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        group_labels[str(g)] = {threshold_to_idx(v) for v in y_threshold[idx]}

    label_to_groups: dict[int, set[str]] = {}
    for g, labels in group_labels.items():
        for lbl in labels:
            label_to_groups.setdefault(lbl, set()).add(g)

    valid_splits: list[tuple[np.ndarray, np.ndarray]] = []
    total_splits = 0

    for train_idx, test_idx in splitter.split(X, y_threshold, groups):
        total_splits += 1
        train_groups = {str(g) for g in groups[train_idx]}
        test_groups = {str(g) for g in groups[test_idx]}

        train_labels = set()
        test_labels = set()
        for g in train_groups:
            train_labels.update(group_labels[g])
        for g in test_groups:
            test_labels.update(group_labels[g])

        ok = True
        for lbl, grp_set in label_to_groups.items():
            if len(grp_set) > 1:
                if lbl not in train_labels or lbl not in test_labels:
                    ok = False
                    break
            else:
                # Only one group has this label: allow it to be evaluated in test
                if lbl not in test_labels:
                    ok = False
                    break

        if ok:
            valid_splits.append((train_idx, test_idx))

    if not valid_splits:
        raise ValueError(
            "No valid group splits satisfy balanced label coverage. "
            "Reduce n_splits or adjust grouping."
        )

    if len(valid_splits) < total_splits:
        print(
            f"  Using {len(valid_splits)}/{total_splits} splits with balanced label coverage"
        )

    return valid_splits


def get_splits_ignore_rare_groups(
    X: np.ndarray,
    y_threshold: np.ndarray,
    groups: np.ndarray,
    splitter,
    seed: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str], list[dict]]:
    """Generate group splits ignoring rare groups, then add them to train/test."""
    from .scoring import threshold_to_idx

    # Map group -> labels and label -> groups
    group_labels: dict[str, set[int]] = {}
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        group_labels[str(g)] = {threshold_to_idx(v) for v in y_threshold[idx]}

    label_to_groups: dict[int, set[str]] = {}
    for g, labels in group_labels.items():
        for lbl in labels:
            label_to_groups.setdefault(lbl, set()).add(g)

    # Rare groups: any group containing a label that appears in only one group
    rare_groups = set()
    for lbl, grp_set in label_to_groups.items():
        if len(grp_set) == 1:
            rare_groups.update(grp_set)

    common_mask = np.array([str(g) not in rare_groups for g in groups])
    common_idx = np.flatnonzero(common_mask)

    if common_idx.size == 0:
        raise ValueError("All groups are rare; cannot create common-group splits.")

    rng = np.random.default_rng(seed)

    base_splits = []
    rare_assignments: list[dict] = []
    for train_idx, test_idx in splitter.split(
        X[common_idx], y_threshold[common_idx], groups[common_idx]
    ):
        # Map back to full indices
        train_full = common_idx[train_idx].tolist()
        test_full = common_idx[test_idx].tolist()

        rare_list = sorted(rare_groups)
        if len(rare_list) == 1:
            # Single rare group goes to test
            g = rare_list[0]
            rare_idx = np.flatnonzero(groups == g).tolist()
            test_full.extend(rare_idx)
            rare_assignments.append({"train": [], "test": [g]})
        elif len(rare_list) > 1:
            rng.shuffle(rare_list)
            # Ensure at least one rare group in each side
            split_point = max(1, len(rare_list) // 2)
            train_rare = rare_list[:split_point]
            test_rare = rare_list[split_point:]
            if not test_rare:
                test_rare = train_rare[-1:]
                train_rare = train_rare[:-1]

            for g in train_rare:
                train_full.extend(np.flatnonzero(groups == g).tolist())
            for g in test_rare:
                test_full.extend(np.flatnonzero(groups == g).tolist())
            rare_assignments.append(
                {"train": list(train_rare), "test": list(test_rare)}
            )
        else:
            rare_assignments.append({"train": [], "test": []})

        base_splits.append((np.array(train_full, dtype=int), np.array(test_full, dtype=int)))

    return base_splits, sorted(rare_groups), rare_assignments


def build_cv_splits(
    X: np.ndarray,
    y_threshold: np.ndarray,
    groups: np.ndarray,
    splitter_or_splits,
    valid_size: float = 0.2,
    seed: int = 42,
) -> list[dict]:
    """Build CV splits with fixed train/val/test indices."""
    if isinstance(splitter_or_splits, list):
        base_splits = splitter_or_splits
    else:
        base_splits = get_valid_group_splits(X, y_threshold, groups, splitter_or_splits)
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
