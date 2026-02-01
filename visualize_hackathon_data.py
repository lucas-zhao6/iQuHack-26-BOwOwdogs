#!/usr/bin/env python3
"""
Extensive visualization of hackathon_public.json.

Generates figures for:
- Circuits: family, n_qubits, source
- Results: status, backend, precision
- Threshold & fidelity: selected threshold/fidelity, sweep curves
- Runtime & memory: wall time, peak RSS, timing decomposition
- Sweep details: timeouts, returncodes, p_return_zero
- Correlations and distributions
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="husl", font_scale=1.0)
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "2026-Quantum-Rings" / "data" / "hackathon_public.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "visualizations"

THRESHOLD_RUNGS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Style
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.2


def load_data():
    """Load JSON and build circuits + full results DataFrames."""
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # Circuits
    circuits = pd.DataFrame([
        {
            "file": c["file"],
            "family": c["family"],
            "n_qubits": c["n_qubits"],
            "source_name": c["source"]["name"],
        }
        for c in data["circuits"]
    ])

    # Results: one row per (file, backend, precision), include all statuses
    result_rows = []
    sweep_rows = []  # one row per (result, threshold) for sweep plots

    for r in data["results"]:
        row = {
            "file": r["file"],
            "backend": r["backend"],
            "precision": r["precision"],
            "status": r.get("status", "unknown"),
        }
        sel = r.get("selection", {})
        row["selected_threshold"] = sel.get("selected_threshold")
        row["selected_fidelity"] = sel.get("selected_mirror_metric_value")
        row["selection_target"] = sel.get("target")
        row["selection_stop_when"] = sel.get("stop_when")

        fwd = r.get("forward", {})
        row["forward_wall_s"] = fwd.get("run_wall_s")
        row["forward_threshold"] = fwd.get("threshold")
        row["forward_shots"] = fwd.get("shots")
        row["forward_unique_outcomes"] = fwd.get("unique_outcomes")
        row["forward_peak_rss_mb"] = fwd.get("peak_rss_mb")
        row["forward_tail_mass"] = fwd.get("tail_mass")

        timing = r.get("forward_timing_estimates", {})
        row["estimated_per_shot_s"] = timing.get("estimated_per_shot_s")
        row["estimated_setup_s"] = timing.get("estimated_setup_s")

        setup = r.get("state_setup", {})
        row["state_setup_wall_s"] = setup.get("run_wall_s")
        row["state_setup_peak_rss_mb"] = setup.get("peak_rss_mb")

        verify = r.get("verify", {})
        row["verify_p_return_zero"] = verify.get("p_return_zero") if verify else None

        result_rows.append(row)

        # Flatten threshold_sweep for sweep-level analysis
        for entry in r.get("threshold_sweep", []):
            sweep_rows.append({
                "file": r["file"],
                "backend": r["backend"],
                "precision": r["precision"],
                "status": r.get("status"),
                "threshold": entry.get("threshold"),
                "sdk_get_fidelity": entry.get("sdk_get_fidelity"),
                "p_return_zero": entry.get("p_return_zero"),
                "run_wall_s": entry.get("run_wall_s"),
                "peak_rss_mb": entry.get("peak_rss_mb"),
                "returncode": entry.get("returncode"),
                "note": entry.get("note", ""),
            })

    results = pd.DataFrame(result_rows)
    sweep = pd.DataFrame(sweep_rows)

    # Join results with circuit metadata
    results = results.merge(
        circuits[["file", "family", "n_qubits", "source_name"]],
        on="file",
        how="left",
    )
    sweep = sweep.merge(
        circuits[["file", "family", "n_qubits"]],
        on="file",
        how="left",
    )

    return data, circuits, results, sweep


def plot_circuits(circuits: pd.DataFrame):
    """Circuit-level visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Count by family
    ax = axes[0, 0]
    fam_counts = circuits["family"].value_counts().sort_values(ascending=True)
    fam_counts.plot.barh(ax=ax)
    ax.set_title("Circuits per family")
    ax.set_xlabel("Count")

    # 2. n_qubits distribution
    ax = axes[0, 1]
    ax.hist(circuits["n_qubits"], bins=min(30, circuits["n_qubits"].nunique()), edgecolor="black", alpha=0.7)
    ax.set_title("Distribution of n_qubits")
    ax.set_xlabel("n_qubits")
    ax.set_ylabel("Count")

    # 3. Source distribution
    ax = axes[1, 0]
    src_counts = circuits["source_name"].value_counts()
    src_counts.plot.pie(ax=ax, autopct="%1.1f%%", startangle=90)
    ax.set_title("Circuits by source")
    ax.set_ylabel("")

    # 4. Family × n_qubits (strip plot)
    ax = axes[1, 1]
    families = circuits["family"].unique()
    for i, fam in enumerate(families):
        subset = circuits[circuits["family"] == fam]
        ax.scatter(subset["n_qubits"], [i] * len(subset), label=fam, alpha=0.8, s=40)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=7)
    ax.set_title("n_qubits by family")
    ax.set_xlabel("n_qubits")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_circuits_overview.png")
    plt.close()


def plot_results_status(results: pd.DataFrame):
    """Result status, backend, precision."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, col, title in [
        (axes[0], "status", "Result status"),
        (axes[1], "backend", "Backend"),
        (axes[2], "precision", "Precision"),
    ]:
        results[col].value_counts().plot.bar(ax=ax)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_results_status_backend_precision.png")
    plt.close()


def plot_selected_threshold_fidelity(results: pd.DataFrame):
    """Selected threshold and fidelity distributions (ok only)."""
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Selected threshold distribution
    ax = axes[0, 0]
    th = ok["selected_threshold"].dropna()
    if not th.empty:
        ax.hist(th.astype(int), bins=range(1, 513), edgecolor="black", alpha=0.7)
        ax.set_xscale("log")
        ax.set_xticks(THRESHOLD_RUNGS)
        ax.set_xticklabels(THRESHOLD_RUNGS)
    ax.set_title("Selected threshold (status=ok)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Count")

    # 2. Selected fidelity distribution
    ax = axes[0, 1]
    fid = ok["selected_fidelity"].dropna()
    if not fid.empty:
        ax.hist(fid, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0.99, color="red", linestyle="--", label="Target 0.99")
    ax.set_title("Selected fidelity (mirror metric)")
    ax.set_xlabel("Fidelity")
    ax.legend()

    # 3. Selected threshold by family (box)
    ax = axes[1, 0]
    ok_th = ok[ok["selected_threshold"].notna()].copy()
    ok_th["selected_threshold"] = ok_th["selected_threshold"].astype(int)
    if not ok_th.empty:
        if HAS_SEABORN:
            sns.boxplot(data=ok_th, x="family", y="selected_threshold", ax=ax)
        else:
            ok_th.boxplot(column="selected_threshold", by="family", ax=ax)
        ax.set_yscale("log")
        ax.set_yticks(THRESHOLD_RUNGS)
        ax.set_yticklabels(THRESHOLD_RUNGS)
        ax.tick_params(axis="x", rotation=70)
    ax.set_title("Selected threshold by family")

    # 4. Selected fidelity by family
    ax = axes[1, 1]
    if not ok.empty and ok["selected_fidelity"].notna().any():
        if HAS_SEABORN:
            sns.boxplot(data=ok, x="family", y="selected_fidelity", ax=ax)
        else:
            ok.boxplot(column="selected_fidelity", by="family", ax=ax)
        ax.axhline(0.99, color="red", linestyle="--", alpha=0.7)
        ax.tick_params(axis="x", rotation=70)
    ax.set_title("Selected fidelity by family")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_selected_threshold_fidelity.png")
    plt.close()


def plot_fidelity_vs_threshold_by_circuit(sweep: pd.DataFrame):
    """Fidelity vs threshold, one line per circuit (file)."""
    if sweep.empty:
        return
    sweep_ok = sweep[sweep["status"] == "ok"].copy()
    sweep_ok = sweep_ok[sweep_ok["note"].fillna("").str.contains("ok")]
    sweep_ok = sweep_ok[sweep_ok["sdk_get_fidelity"].notna()]
    if sweep_ok.empty:
        return

    # Mean fidelity per (file, threshold) so one line per circuit
    by_circuit = sweep_ok.groupby(["file", "threshold"])["sdk_get_fidelity"].mean().reset_index()
    circuits = by_circuit["file"].unique()

    fig, ax = plt.subplots(figsize=(12, 8))
    n = len(circuits)
    cmap = matplotlib.colormaps.get_cmap("nipy_spectral")
    for i, file in enumerate(circuits):
        sub = by_circuit[by_circuit["file"] == file].sort_values("threshold")
        if sub.empty:
            continue
        color = cmap(i / max(n - 1, 1))
        ax.plot(sub["threshold"], sub["sdk_get_fidelity"], "-o", label=file, markersize=3, color=color, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xticks(THRESHOLD_RUNGS)
    ax.set_xticklabels(THRESHOLD_RUNGS)
    ax.axhline(0.99, color="red", linestyle="--", alpha=0.8, label="Target 0.99")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("sdk_get_fidelity (mean over backend×precision)")
    ax.set_title("Fidelity vs threshold by circuit")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_fidelity_vs_threshold_by_circuit.png", bbox_inches="tight")
    plt.close()


def _config_has_fidelity_drop(sweep_sub: pd.DataFrame) -> bool:
    """True if fidelity ever decreases when threshold increases (same config)."""
    sub = sweep_sub.sort_values("threshold")
    fids = sub["sdk_get_fidelity"].values
    for i in range(len(fids) - 1):
        a, b = fids[i], fids[i + 1]
        if pd.notna(a) and pd.notna(b) and b < a:
            return True
    return False


def plot_fidelity_drops_anomaly(sweep: pd.DataFrame):
    """Fidelity vs threshold for configs where fidelity drops at higher threshold (one line per config)."""
    if sweep.empty:
        return
    sweep_ok = sweep[sweep["status"] == "ok"].copy()
    sweep_ok = sweep_ok[sweep_ok["note"].fillna("").str.contains("ok")]
    sweep_ok = sweep_ok[sweep_ok["sdk_get_fidelity"].notna()]
    if sweep_ok.empty:
        return

    # Configs that have at least one drop
    configs_with_drop = []
    for (file, backend, precision), grp in sweep_ok.groupby(["file", "backend", "precision"]):
        if _config_has_fidelity_drop(grp):
            configs_with_drop.append((file, backend, precision))

    if not configs_with_drop:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    n = len(configs_with_drop)
    cmap = matplotlib.colormaps.get_cmap("nipy_spectral")
    for i, (file, backend, precision) in enumerate(configs_with_drop):
        sub = sweep_ok[
            (sweep_ok["file"] == file)
            & (sweep_ok["backend"] == backend)
            & (sweep_ok["precision"] == precision)
        ].sort_values("threshold")
        if sub.empty:
            continue
        label = f"{file} | {backend}/{precision}"
        color = cmap(i / max(n - 1, 1))
        ax.plot(
            sub["threshold"],
            sub["sdk_get_fidelity"],
            "-o",
            label=label,
            markersize=4,
            color=color,
            alpha=0.9,
        )
    ax.set_xscale("log")
    ax.set_xticks(THRESHOLD_RUNGS)
    ax.set_xticklabels(THRESHOLD_RUNGS)
    ax.axhline(0.99, color="red", linestyle="--", alpha=0.8, label="Target 0.99")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("sdk_get_fidelity")
    ax.set_title("Fidelity vs threshold — configs where fidelity drops at higher threshold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04b_fidelity_drops_anomaly.png", bbox_inches="tight")
    plt.close()


def plot_sweep_curves(results: pd.DataFrame, sweep: pd.DataFrame):
    """Threshold sweep: fidelity and runtime vs threshold."""
    ok = results[results["status"] == "ok"]
    if ok.empty or sweep.empty:
        return

    # Aggregate sweep: mean fidelity and run_wall_s per (threshold, family)
    sweep_ok = sweep[sweep["status"] == "ok"].copy()
    sweep_ok = sweep_ok[sweep_ok["note"].str.contains("ok", na=True)]
    if sweep_ok.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Fidelity vs threshold by family (mean)
    ax = axes[0]
    for fam in sweep_ok["family"].dropna().unique():
        sub = sweep_ok[sweep_ok["family"] == fam]
        agg = sub.groupby("threshold")["sdk_get_fidelity"].mean()
        if agg.notna().any():
            ax.plot(agg.index, agg.values, "-o", label=fam, markersize=4)
    ax.set_xscale("log")
    ax.set_xticks(THRESHOLD_RUNGS)
    ax.set_xticklabels(THRESHOLD_RUNGS)
    ax.axhline(0.99, color="gray", linestyle="--", alpha=0.8)
    ax.set_title("Mean fidelity vs threshold (by family)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("sdk_get_fidelity")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Run wall time vs threshold by family (mean)
    ax = axes[1]
    for fam in sweep_ok["family"].dropna().unique()[:12]:  # limit for readability
        sub = sweep_ok[sweep_ok["family"] == fam]
        agg = sub.groupby("threshold")["run_wall_s"].mean()
        if agg.notna().any():
            ax.plot(agg.index, agg.values, "-o", label=fam, markersize=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(THRESHOLD_RUNGS)
    ax.set_xticklabels(THRESHOLD_RUNGS)
    ax.set_title("Mean run_wall_s vs threshold (by family, subset)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("run_wall_s")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_sweep_fidelity_runtime_curves.png")
    plt.close()


def plot_runtime_memory(results: pd.DataFrame):
    """Forward runtime and memory (ok only)."""
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. forward_wall_s distribution
    ax = axes[0, 0]
    w = ok["forward_wall_s"].dropna()
    if not w.empty:
        ax.hist(w, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("forward_wall_s")
    ax.set_title("Forward run wall time")

    # 2. forward_wall_s by family (box)
    ax = axes[0, 1]
    if ok["forward_wall_s"].notna().any():
        if HAS_SEABORN:
            sns.boxplot(data=ok, x="family", y="forward_wall_s", ax=ax)
        else:
            ok.boxplot(column="forward_wall_s", by="family", ax=ax)
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=70)
    ax.set_title("Forward wall_s by family")

    # 3. forward_peak_rss_mb distribution
    ax = axes[1, 0]
    rss = ok["forward_peak_rss_mb"].dropna()
    if not rss.empty:
        ax.hist(rss, bins=30, edgecolor="black", alpha=0.7)
    ax.set_title("Forward peak RSS (MB)")
    ax.set_xlabel("peak_rss_mb")

    # 4. forward_wall_s vs n_qubits (scatter by family)
    ax = axes[1, 1]
    sub = ok[ok["forward_wall_s"].notna() & ok["n_qubits"].notna()]
    if not sub.empty:
        for fam in sub["family"].unique()[:15]:
            s = sub[sub["family"] == fam]
            ax.scatter(s["n_qubits"], s["forward_wall_s"], label=fam, alpha=0.7, s=30)
        ax.set_yscale("log")
        ax.set_xlabel("n_qubits")
        ax.set_ylabel("forward_wall_s")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6)
    ax.set_title("Forward wall_s vs n_qubits")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_runtime_memory.png")
    plt.close()


def plot_timing_decomposition(results: pd.DataFrame):
    """Estimated setup vs per-shot time (ok only)."""
    ok = results[results["status"] == "ok"].copy()
    ok = ok[ok["estimated_setup_s"].notna() & ok["estimated_per_shot_s"].notna()]
    if ok.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sc = ax.scatter(ok["estimated_setup_s"], ok["estimated_per_shot_s"], c=ok["n_qubits"], cmap="viridis", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("estimated_setup_s")
    ax.set_ylabel("estimated_per_shot_s")
    ax.set_title("Timing decomposition (color = n_qubits)")
    plt.colorbar(sc, ax=ax, label="n_qubits")

    ax = axes[1]
    ok["total_estimated_10k"] = ok["estimated_setup_s"] + 10000 * ok["estimated_per_shot_s"]
    ax.scatter(ok["forward_wall_s"], ok["total_estimated_10k"], alpha=0.7)
    ax.plot([1, 1e5], [1, 1e5], "k--", alpha=0.5, label="y=x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual forward_wall_s")
    ax.set_ylabel("Estimated setup + 10k*per_shot")
    ax.set_title("Actual vs estimated 10k-shot time")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_timing_decomposition.png")
    plt.close()


def plot_sweep_errors(sweep: pd.DataFrame):
    """Sweep returncodes, timeouts, notes."""
    if sweep.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 1. Returncode distribution in sweep
    ax = axes[0]
    rc = sweep["returncode"].value_counts().sort_index()
    rc.plot.bar(ax=ax)
    ax.set_title("Sweep returncode distribution")
    ax.set_xlabel("returncode")

    # 2. Note distribution (e.g. ok vs mirror_timeout)
    ax = axes[1]
    note = sweep["note"].fillna("(null)").value_counts()
    note.plot.bar(ax=ax)
    ax.set_title("Sweep note distribution")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_sweep_errors.png")
    plt.close()


def plot_timeouts_by_threshold(sweep: pd.DataFrame):
    """Count of timeouts (or non-ok) per threshold."""
    if sweep.empty:
        return

    timeout = sweep["note"].fillna("").str.contains("timeout", case=False)
    sweep_copy = sweep.copy()
    sweep_copy["is_timeout"] = timeout

    agg = sweep_copy.groupby("threshold").agg(
        total=("file", "count"),
        timeouts=("is_timeout", "sum"),
    ).reset_index()
    agg["ok_count"] = agg["total"] - agg["timeouts"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(agg))
    ax.bar([i - 0.2 for i in x], agg["ok_count"], width=0.4, label="ok")
    ax.bar([i + 0.2 for i in x], agg["timeouts"], width=0.4, label="timeout/non-ok")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["threshold"])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Count")
    ax.set_title("Sweep outcomes: ok vs timeout by threshold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_timeouts_by_threshold.png")
    plt.close()


def plot_p_return_zero(sweep: pd.DataFrame, results: pd.DataFrame):
    """p_return_zero in sweep and verify."""
    ok = results[results["status"] == "ok"]
    if ok.empty and sweep.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Verify p_return_zero
    if not ok.empty and ok["verify_p_return_zero"].notna().any():
        ax = axes[0]
        ax.hist(ok["verify_p_return_zero"].dropna(), bins=20, edgecolor="black", alpha=0.7)
        ax.set_title("Verify p_return_zero (status=ok)")
        ax.set_xlabel("p_return_zero")

    # Sweep p_return_zero at selected thresholds (sample)
    sub = sweep[sweep["p_return_zero"].notna()]
    if not sub.empty:
        ax = axes[1]
        sub_sample = sub.groupby(["file", "backend", "precision", "threshold"]).first().reset_index()
        # Show distribution of p_return_zero per threshold
        for th in [1, 8, 16, 32]:
            s = sub_sample[sub_sample["threshold"] == th]["p_return_zero"]
            if not s.empty:
                ax.hist(s, bins=15, alpha=0.5, label=f"threshold={th}")
        ax.set_title("Sweep p_return_zero by threshold (sample)")
        ax.set_xlabel("p_return_zero")
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_p_return_zero.png")
    plt.close()


def plot_forward_extra(results: pd.DataFrame):
    """Forward unique_outcomes, tail_mass."""
    ok = results[results["status"] == "ok"]
    if ok.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if ok["forward_unique_outcomes"].notna().any():
        ax = axes[0]
        ax.hist(ok["forward_unique_outcomes"].dropna(), bins=30, edgecolor="black", alpha=0.7)
        ax.set_title("Forward unique_outcomes")
        ax.set_xlabel("unique_outcomes")

    if ok["forward_tail_mass"].notna().any():
        ax = axes[1]
        ax.hist(ok["forward_tail_mass"].dropna(), bins=20, edgecolor="black", alpha=0.7)
        ax.set_title("Forward tail_mass")
        ax.set_xlabel("tail_mass")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_forward_extra.png")
    plt.close()


def plot_correlations(results: pd.DataFrame):
    """Numeric correlation heatmap (ok only)."""
    ok = results[results["status"] == "ok"]
    num_cols = [
        "n_qubits",
        "selected_threshold",
        "selected_fidelity",
        "forward_wall_s",
        "forward_peak_rss_mb",
        "estimated_setup_s",
        "estimated_per_shot_s",
        "state_setup_wall_s",
        "state_setup_peak_rss_mb",
        "verify_p_return_zero",
    ]
    present = [c for c in num_cols if c in ok.columns and ok[c].notna().any()]
    if len(present) < 2:
        return

    corr = ok[present].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    if HAS_SEABORN:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
    else:
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(present)))
        ax.set_yticks(range(len(present)))
        ax.set_xticklabels(present, rotation=45, ha="right")
        ax.set_yticklabels(present)
        for i in range(len(present)):
            for j in range(len(present)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax)
    ax.set_title("Correlation matrix (status=ok)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "11_correlations.png")
    plt.close()


def plot_backend_precision_breakdown(results: pd.DataFrame):
    """Metrics by backend and precision (ok only)."""
    ok = results[results["status"] == "ok"]
    if ok.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, col, title in [
        (axes[0, 0], "selected_threshold", "Selected threshold by backend"),
        (axes[0, 1], "forward_wall_s", "Forward wall_s by backend"),
        (axes[1, 0], "selected_threshold", "Selected threshold by precision"),
        (axes[1, 1], "forward_wall_s", "Forward wall_s by precision"),
    ]:
        xcol = "backend" if "backend" in title else "precision"
        if HAS_SEABORN:
            sns.boxplot(data=ok, x=xcol, y=col, ax=ax)
        else:
            ok.boxplot(column=col, by=xcol, ax=ax)
        if col == "forward_wall_s" or (ok[col].notna().any() and ok[col].max() > 100):
            ax.set_yscale("log")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "12_backend_precision_breakdown.png")
    plt.close()


def plot_n_qubits_vs_metrics(results: pd.DataFrame):
    """n_qubits vs selected_threshold, forward_wall_s, peak_rss (ok only)."""
    ok = results[results["status"] == "ok"]
    if ok.empty or ok["n_qubits"].isna().all():
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, ycol, ylabel, logy in [
        (axes[0], "selected_threshold", "Selected threshold", True),
        (axes[1], "forward_wall_s", "Forward wall_s", True),
        (axes[2], "forward_peak_rss_mb", "Peak RSS (MB)", True),
    ]:
        if ycol not in ok.columns or ok[ycol].isna().all():
            continue
        ax.scatter(ok["n_qubits"], ok[ycol], alpha=0.6, s=40)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("n_qubits")
        ax.set_ylabel(ylabel)
        ax.set_title(f"n_qubits vs {ycol}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "13_n_qubits_vs_metrics.png")
    plt.close()


def plot_results_per_circuit(results: pd.DataFrame, circuits: pd.DataFrame):
    """Number of result rows (configs) per circuit file."""
    counts = results.groupby("file").size().reset_index(name="n_results")
    merged = circuits.merge(counts, on="file", how="left").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    merged_sorted = merged.sort_values("n_results", ascending=False)
    ax.barh(range(len(merged_sorted)), merged_sorted["n_results"])
    ax.set_yticks(range(len(merged_sorted)))
    ax.set_yticklabels(merged_sorted["file"], fontsize=7)
    ax.set_xlabel("Number of result configs (backend × precision)")
    ax.set_title("Result configs per circuit")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "14_results_per_circuit.png")
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    data, circuits, results, sweep = load_data()
    print(f"Circuits: {len(circuits)}, Results: {len(results)}, Sweep rows: {len(sweep)}")

    print("Plotting circuits...")
    plot_circuits(circuits)
    print("Plotting result status/backend/precision...")
    plot_results_status(results)
    print("Plotting selected threshold/fidelity...")
    plot_selected_threshold_fidelity(results)
    print("Plotting fidelity vs threshold by circuit...")
    plot_fidelity_vs_threshold_by_circuit(sweep)
    print("Plotting fidelity-drops anomaly (per-config)...")
    plot_fidelity_drops_anomaly(sweep)
    print("Plotting sweep curves...")
    plot_sweep_curves(results, sweep)
    print("Plotting runtime/memory...")
    plot_runtime_memory(results)
    print("Plotting timing decomposition...")
    plot_timing_decomposition(results)
    print("Plotting sweep errors...")
    plot_sweep_errors(sweep)
    print("Plotting timeouts by threshold...")
    plot_timeouts_by_threshold(sweep)
    print("Plotting p_return_zero...")
    plot_p_return_zero(sweep, results)
    print("Plotting forward extra...")
    plot_forward_extra(results)
    print("Plotting correlations...")
    plot_correlations(results)
    print("Plotting backend/precision breakdown...")
    plot_backend_precision_breakdown(results)
    print("Plotting n_qubits vs metrics...")
    plot_n_qubits_vs_metrics(results)
    print("Plotting results per circuit...")
    plot_results_per_circuit(results, circuits)

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
    sys.exit(0)
