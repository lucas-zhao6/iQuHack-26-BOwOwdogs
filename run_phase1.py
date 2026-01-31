#!/usr/bin/env python3
"""
Phase 1: Data Pipeline & Feature Engineering

Runs the full feature extraction pipeline on the training data and produces:
1. A feature DataFrame saved as CSV and pickle
2. Summary statistics and correlation analysis
3. Family classification accuracy report
4. Key feature-target correlations for threshold and runtime
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_pipeline import build_full_training_dataframe
from src.family_classifier import FamilyClassifierKNN
from src.feature_selection import FeatureSelector

# Paths
CIRCUITS_DIR = PROJECT_ROOT / "2026-Quantum-Rings" / "circuits"
TRAINING_JSON = PROJECT_ROOT / "2026-Quantum-Rings" / "data" / "hackathon_public.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Step 1: Build full training dataframe ──────────────────────────
    print("=" * 70)
    print("PHASE 1: Feature Extraction Pipeline")
    print("=" * 70)

    print("\n[1/5] Building full training dataframe...")
    df = build_full_training_dataframe(CIRCUITS_DIR, TRAINING_JSON)
    print(f"  Shape: {df.shape}")
    print(f"  Circuits: {df['file'].nunique()}")
    print(f"  Configs: {len(df)}")

    # Save
    df.to_csv(OUTPUT_DIR / "training_features.csv", index=False)
    df.to_pickle(OUTPUT_DIR / "training_features.pkl")
    print(f"  Saved to outputs/training_features.csv")

    # ── Step 2: Feature summary ────────────────────────────────────────
    print("\n[2/5] Feature summary...")

    # Identify feature columns (exclude labels and metadata)
    meta_cols = [
        "file", "backend", "precision", "true_family", "predicted_family",
        "selected_threshold", "selected_fidelity", "selected_threshold_log2",
        "forward_wall_s", "forward_threshold", "forward_unique_outcomes",
        "forward_peak_rss_mb", "estimated_per_shot_s", "estimated_setup_s",
        "state_setup_wall_s", "state_setup_peak_rss_mb",
        "log_forward_wall_s", "log_setup_s", "log_per_shot_s",
        "is_gpu", "is_double",
        "verify_p_return_zero",
        "sweep_min_fidelity", "sweep_max_fidelity", "n_sweep_rungs",
        "fidelity_at_1", "rungs_to_099", "biggest_fid_jump", "biggest_jump_rung",
    ]
    sweep_cols = [c for c in df.columns if c.startswith("sweep_")]
    exclude = set(meta_cols + sweep_cols)
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]

    print(f"  Total feature columns: {len(feature_cols)}")
    print(f"  Feature list:")
    for i, f in enumerate(sorted(feature_cols), 1):
        print(f"    {i:3d}. {f}")

    # ── Step 3: Family classification accuracy ─────────────────────────
    print("\n[3/5] Family classification accuracy...")

    # Deduplicate to circuit level for family comparison
    circuit_df = df.drop_duplicates(subset="file")[["file", "true_family", "predicted_family"]].copy()

    n_correct = (circuit_df["true_family"] == circuit_df["predicted_family"]).sum()
    n_total = len(circuit_df)
    print(f"  Rule-based accuracy: {n_correct}/{n_total} ({100 * n_correct / n_total:.1f}%)")

    mismatches = circuit_df[circuit_df["true_family"] != circuit_df["predicted_family"]]
    if len(mismatches) > 0:
        print(f"  Misclassified:")
        for _, row in mismatches.iterrows():
            print(f"    {row['file']}: true={row['true_family']}, pred={row['predicted_family']}")
    else:
        print(f"  All circuits classified correctly!")

    # Train KNN classifier and evaluate with leave-one-out
    print("\n  KNN classifier (leave-one-out cross-validation):")
    knn_features = [c for c in feature_cols if not c.startswith("gate_") or c.endswith("_frac")]
    knn_feature_cols = [c for c in knn_features if c in df.columns]

    circuit_feature_dicts = []
    circuit_labels = []
    for _, row in circuit_df.iterrows():
        file_rows = df[df["file"] == row["file"]].iloc[0]
        feat_dict = {f: float(file_rows[f]) if pd.notna(file_rows[f]) else 0.0 for f in knn_feature_cols}
        circuit_feature_dicts.append(feat_dict)
        circuit_labels.append(row["true_family"])

    # Leave-one-out CV
    knn_correct = 0
    for i in range(len(circuit_feature_dicts)):
        train_feats = circuit_feature_dicts[:i] + circuit_feature_dicts[i + 1:]
        train_labels = circuit_labels[:i] + circuit_labels[i + 1:]
        test_feat = circuit_feature_dicts[i]

        clf = FamilyClassifierKNN(n_neighbors=min(3, len(train_feats)))
        clf.fit(train_feats, train_labels)
        pred = clf.predict(test_feat)
        if pred == circuit_labels[i]:
            knn_correct += 1

    print(f"  KNN LOO accuracy: {knn_correct}/{n_total} ({100 * knn_correct / n_total:.1f}%)")

    # ── Step 4: Key correlations with targets ──────────────────────────
    print("\n[4/5] Top feature correlations with targets...")

    numeric_df = df[feature_cols + ["selected_threshold_log2", "log_forward_wall_s"]].copy()
    numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")

    # Threshold correlations
    if "selected_threshold_log2" in numeric_df.columns:
        thr_corr = numeric_df[feature_cols].corrwith(
            numeric_df["selected_threshold_log2"]
        ).dropna().abs().sort_values(ascending=False)

        print("\n  Top 15 features correlated with threshold (|r|):")
        for feat, corr in thr_corr.head(15).items():
            print(f"    {corr:.3f}  {feat}")

    # Runtime correlations
    if "log_forward_wall_s" in numeric_df.columns:
        rt_corr = numeric_df[feature_cols].corrwith(
            numeric_df["log_forward_wall_s"]
        ).dropna().abs().sort_values(ascending=False)

        print("\n  Top 15 features correlated with log(runtime) (|r|):")
        for feat, corr in rt_corr.head(15).items():
            print(f"    {corr:.3f}  {feat}")

    # FeatureSelector preview (matches Phase 2 selection logic)
    print("\n  FeatureSelector (split threshold/runtime) selection:")
    valid_mask = df["selected_threshold"].notna() & df["forward_wall_s"].notna()
    fs_df = df[valid_mask].copy()
    X_fs = fs_df[feature_cols].copy()
    X_fs = X_fs.fillna(X_fs.mean())
    y_thr_fs = fs_df["selected_threshold"].values.astype(float)
    y_rt_fs = fs_df["forward_wall_s"].values.astype(float)

    selector = FeatureSelector(k=30)
    selector.fit(X_fs.values, y_thr_fs, y_rt_fs, feature_cols)
    selected_thr = selector.get_selected_names(target="threshold")
    selected_rt = selector.get_selected_names(target="runtime")

    print("    Threshold features:")
    for i, name in enumerate(selected_thr, 1):
        print(f"      {i:3d}. {name}")

    print("    Runtime features:")
    for i, name in enumerate(selected_rt, 1):
        print(f"      {i:3d}. {name}")

    # ── Step 5: Per-family threshold summary ───────────────────────────
    print("\n[5/5] Per-family threshold distribution...")

    family_summary = df.groupby("true_family").agg(
        n_circuits=("file", "nunique"),
        mean_threshold=("selected_threshold", "mean"),
        min_threshold=("selected_threshold", "min"),
        max_threshold=("selected_threshold", "max"),
        mean_runtime=("forward_wall_s", "mean"),
        mean_max_cut_pressure=("max_cut_pressure", "mean"),
    ).sort_values("mean_threshold", ascending=False)

    print(f"\n  {'Family':<25s} {'Circuits':>8s} {'Mean Thr':>10s} {'Range':>12s} {'Mean RT':>10s} {'MaxCP':>8s}")
    print("  " + "-" * 75)
    for family, row in family_summary.iterrows():
        print(
            f"  {family:<25s} {int(row['n_circuits']):>8d} "
            f"{row['mean_threshold']:>10.1f} "
            f"{int(row['min_threshold']):>4d}-{int(row['max_threshold']):<6d} "
            f"{row['mean_runtime']:>10.1f} "
            f"{row['mean_max_cut_pressure']:>8.1f}"
        )

    # ── Done ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 1 COMPLETE")
    print(f"  Feature DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Circuit features extracted: {len(feature_cols)}")
    print(f"  Family classification: {n_correct}/{n_total} correct")
    print(f"  Output saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
