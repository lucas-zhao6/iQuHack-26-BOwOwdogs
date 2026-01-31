"""
Feature pipeline: end-to-end extraction of all features from circuits + training data.

Combines QASM parsing, MPS features, graph features, gate fingerprints, and
training labels into a single unified DataFrame ready for model training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from .qasm_parser import parse_qasm, ParsedCircuit
from .mps_features import (
    compute_cut_pressure,
    compute_gate_span_features,
    compute_depth_features,
    compute_entanglement_layer_features,
    compute_linear_cut_metrics,
)
from .graph_features import (
    compute_graph_features,
    compute_gate_type_fingerprint,
    compute_param_stats,
)
from .family_classifier import classify_family_combined, FamilyClassifierKNN
from .data_loader import (
    load_training_data,
    build_circuit_metadata,
    build_results_dataframe,
)

# Feature toggles
ENABLE_EXPANSION_STRUCTURE_FEATURES = False
ENABLE_MPS_LINEAR_CUT_FEATURES = True


def extract_circuit_features(circuit: ParsedCircuit) -> Dict[str, Any]:
    """
    Extract ALL features from a single parsed circuit.

    Returns a flat dict of ~80+ features combining:
    - Basic metadata (n_qubits, n_regs, total_gates)
    - Gate type fingerprint (counts and fractions for each gate type)
    - MPS cut-pressure features
    - Gate span features
    - Depth features
    - Entanglement layer features
    - Graph connectivity features
    - Gate parameter statistics
    - Family classification
    """
    features: Dict[str, Any] = {}

    # Basic metadata
    features["n_qubits"] = circuit.total_qubits
    features["n_cbits"] = circuit.total_cbits
    features["n_qregs"] = circuit.n_qregs
    features["n_cregs"] = circuit.n_cregs

    # Gate type fingerprint (~40 features)
    gate_fp = compute_gate_type_fingerprint(circuit)
    features.update(gate_fp)

    # MPS cut-pressure (~7 features)
    cut_pressure = compute_cut_pressure(circuit)
    features.update(cut_pressure)

    # Linear-order cut metrics (~2 features)
    if ENABLE_MPS_LINEAR_CUT_FEATURES:
        linear_cut = compute_linear_cut_metrics(circuit)
        features.update(linear_cut)

    # Gate span features (~5 features)
    span_feats = compute_gate_span_features(circuit)
    features.update(span_feats)

    # Depth features (~3 features)
    depth_feats = compute_depth_features(circuit)
    features.update(depth_feats)

    # Entangling layer features (~3 features)
    ent_feats = compute_entanglement_layer_features(circuit)
    features.update(ent_feats)

    # Graph features (~16 features)
    graph_feats = compute_graph_features(
        circuit,
        include_expansion_structure=ENABLE_EXPANSION_STRUCTURE_FEATURES,
    )
    features.update(graph_feats)

    # Parameter statistics (~6 features)
    param_feats = compute_param_stats(circuit)
    features.update(param_feats)

    # Cross-feature interactions (key derived features)
    n_q = max(circuit.total_qubits, 1)
    features["gates_per_qubit"] = features["total_gates"] / n_q
    features["entangling_gates_per_qubit"] = features["n_entangling_gates"] / n_q
    features["cut_pressure_per_qubit"] = features["max_cut_pressure"] / n_q
    features["log_total_gates"] = np.log1p(features["total_gates"])
    features["log_n_qubits"] = np.log1p(circuit.total_qubits)
    features["log_max_cut_pressure"] = np.log1p(features["max_cut_pressure"])

    # Family classification
    predicted_family = classify_family_combined(circuit, graph_feats, gate_fp)
    features["predicted_family"] = predicted_family

    return features


def build_full_training_dataframe(
    circuits_dir: str | Path,
    training_json: str | Path,
) -> pd.DataFrame:
    """
    Build the complete training DataFrame by:
    1. Parsing all QASM circuits and extracting features
    2. Loading training results (labels + sweep data)
    3. Merging on circuit filename

    Returns a DataFrame where each row is one (circuit, backend, precision) config
    with all features + labels.
    """
    circuits_dir = Path(circuits_dir)
    training_json = Path(training_json)

    # Load training data
    raw_data = load_training_data(training_json)
    circuit_meta = build_circuit_metadata(raw_data)
    results_df = build_results_dataframe(raw_data)

    # Parse each circuit and extract features
    circuit_features_rows = []
    for _, row in circuit_meta.iterrows():
        qasm_path = circuits_dir / row["file"]
        if not qasm_path.exists():
            print(f"  WARNING: {qasm_path} not found, skipping")
            continue

        circuit = parse_qasm(qasm_path)
        feats = extract_circuit_features(circuit)
        feats["file"] = row["file"]
        feats["true_family"] = row["family"]
        circuit_features_rows.append(feats)

    circuit_features_df = pd.DataFrame(circuit_features_rows)

    # Merge circuit features with results (training labels)
    merged = results_df.merge(circuit_features_df, on="file", how="left")

    # Add encoded backend/precision columns
    merged["is_gpu"] = (merged["backend"] == "GPU").astype(float)
    merged["is_double"] = (merged["precision"] == "double").astype(float)

    # Log-transform runtime targets
    if "forward_wall_s" in merged.columns:
        merged["log_forward_wall_s"] = np.log1p(merged["forward_wall_s"].fillna(0))
    if "estimated_setup_s" in merged.columns:
        merged["log_setup_s"] = np.log1p(merged["estimated_setup_s"].fillna(0))
    if "estimated_per_shot_s" in merged.columns:
        merged["log_per_shot_s"] = np.log1p(merged["estimated_per_shot_s"].fillna(0))

    return merged


def extract_holdout_circuit_features(
    qasm_path: str | Path,
) -> Dict[str, Any]:
    """Extract features for a single holdout circuit (no labels)."""
    circuit = parse_qasm(qasm_path)
    return extract_circuit_features(circuit)
