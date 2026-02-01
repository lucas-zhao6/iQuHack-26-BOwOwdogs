"""
MPS (Matrix Product State) physics-informed feature extraction.

Computes features that directly relate to the difficulty of simulating
a quantum circuit with a tensor-network / MPS simulator, where the
'threshold' parameter controls truncation of singular values (bond dimension).
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np

from .qasm_parser import ParsedCircuit, TWO_QUBIT_GATES, THREE_QUBIT_GATES


def _get_two_qubit_edges(circuit: ParsedCircuit) -> List[Tuple[int, int]]:
    """Extract all two-qubit gate interactions as (q_i, q_j) pairs."""
    edges = []
    for gate, gidx in zip(circuit.gates, circuit.gate_global_indices):
        if gate.name in TWO_QUBIT_GATES and len(gidx) >= 2:
            edges.append((gidx[0], gidx[1]))
        elif gate.name in THREE_QUBIT_GATES and len(gidx) >= 2:
            # For 3-qubit gates, add all pairwise interactions
            for i in range(len(gidx)):
                for j in range(i + 1, len(gidx)):
                    edges.append((gidx[i], gidx[j]))
    return edges


def compute_cut_pressure(circuit: ParsedCircuit) -> Dict[str, float]:
    """
    Compute the MPS cut-pressure profile.

    For a linear qubit ordering 0..n-1, at each bond position k (between
    qubit k-1 and k), count how many two-qubit gates span across that cut.
    A gate on qubits (i, j) with i < j crosses all cuts at positions
    i+1, i+2, ..., j.

    The maximum cut-pressure is a direct proxy for the maximum bond dimension
    the MPS simulator needs — which is what the threshold controls.

    Returns dict of:
        max_cut_pressure: maximum across all bond positions
        mean_cut_pressure: average cut pressure
        std_cut_pressure: standard deviation
        sum_cut_pressure: total (related to total entanglement generated)
        cut_pressure_entropy: Shannon entropy of normalized pressure profile
        n_active_cuts: number of bond positions with nonzero pressure
    """
    n = circuit.total_qubits
    if n <= 1:
        return {
            "max_cut_pressure": 0.0,
            "mean_cut_pressure": 0.0,
            "std_cut_pressure": 0.0,
            "sum_cut_pressure": 0.0,
            "cut_pressure_entropy": 0.0,
            "n_active_cuts": 0,
            "cut_pressure_frac_active": 0.0,
        }

    edges = _get_two_qubit_edges(circuit)
    pressure = np.zeros(n, dtype=np.float64)

    for q_i, q_j in edges:
        lo, hi = min(q_i, q_j), max(q_i, q_j)
        # Gate crosses cuts at positions lo+1 through hi
        for cut in range(lo + 1, hi + 1):
            pressure[cut] += 1

    # Cuts are at positions 1..n-1 (between qubit k-1 and k)
    bond_pressure = pressure[1:]  # positions 1..n-1

    max_cp = float(bond_pressure.max()) if len(bond_pressure) > 0 else 0.0
    mean_cp = float(bond_pressure.mean()) if len(bond_pressure) > 0 else 0.0
    std_cp = float(bond_pressure.std()) if len(bond_pressure) > 0 else 0.0
    sum_cp = float(bond_pressure.sum())
    n_active = int(np.count_nonzero(bond_pressure))

    # Shannon entropy of normalized pressure profile
    if sum_cp > 0:
        p_norm = bond_pressure / sum_cp
        p_norm = p_norm[p_norm > 0]
        entropy = float(-np.sum(p_norm * np.log2(p_norm)))
    else:
        entropy = 0.0

    return {
        "max_cut_pressure": max_cp,
        "mean_cut_pressure": mean_cp,
        "std_cut_pressure": std_cp,
        "sum_cut_pressure": sum_cp,
        "cut_pressure_entropy": entropy,
        "n_active_cuts": n_active,
        "cut_pressure_frac_active": n_active / max(len(bond_pressure), 1),
    }


def compute_linear_cut_metrics(circuit: ParsedCircuit) -> Dict[str, float]:
    """
    Compute linear-order cut metrics for tensor-network/MPS intuition.

    - edge_cutwidth_natural: max # of UNIQUE interaction edges crossing a cut
    - entangling_cutwidth_natural: max # of entangling gates crossing a cut
      (counts multiplicity of gates)
    """
    n = circuit.total_qubits
    if n <= 1:
        return {
            "edge_cutwidth_natural": 0.0,
            "entangling_cutwidth_natural": 0.0,
        }

    edges = _get_two_qubit_edges(circuit)
    if not edges:
        return {
            "edge_cutwidth_natural": 0.0,
            "entangling_cutwidth_natural": 0.0,
        }

    unique_edges = {(min(i, j), max(i, j)) for i, j in edges}

    max_edge_cut = 0
    max_gate_cut = 0
    for cut in range(1, n):
        edge_cross = 0
        gate_cross = 0
        for i, j in unique_edges:
            if i < cut <= j:
                edge_cross += 1
        for i, j in edges:
            lo, hi = (i, j) if i <= j else (j, i)
            if lo < cut <= hi:
                gate_cross += 1
        if edge_cross > max_edge_cut:
            max_edge_cut = edge_cross
        if gate_cross > max_gate_cut:
            max_gate_cut = gate_cross

    return {
        "edge_cutwidth_natural": float(max_edge_cut),
        "entangling_cutwidth_natural": float(max_gate_cut),
    }


def compute_gate_span_features(circuit: ParsedCircuit) -> Dict[str, float]:
    """
    Compute statistics about the 'span' of two-qubit gates.

    Span = |q_i - q_j| for a gate on qubits (i, j).
    Long-range gates are harder for MPS because they create entanglement
    across many bonds simultaneously.
    """
    edges = _get_two_qubit_edges(circuit)

    if not edges:
        return {
            "max_gate_span": 0.0,
            "mean_gate_span": 0.0,
            "std_gate_span": 0.0,
            "frac_nearest_neighbor": 1.0,
            "frac_long_range": 0.0,
        }

    spans = [abs(q_i - q_j) for q_i, q_j in edges]
    spans_arr = np.array(spans, dtype=np.float64)

    nn_count = sum(1 for s in spans if s == 1)

    return {
        "max_gate_span": float(spans_arr.max()),
        "mean_gate_span": float(spans_arr.mean()),
        "std_gate_span": float(spans_arr.std()),
        "frac_nearest_neighbor": nn_count / len(spans),
        "frac_long_range": sum(1 for s in spans if s > circuit.total_qubits // 4) / len(spans),
    }


def compute_depth_features(circuit: ParsedCircuit) -> Dict[str, float]:
    """
    Estimate circuit depth and layer structure.

    Simulates a greedy ASAP scheduler: for each gate, find the earliest
    time step where all its qubits are free.
    """
    n = circuit.total_qubits
    qubit_time = [0] * n  # next available time for each qubit

    depths = []
    for gate, gidx in zip(circuit.gates, circuit.gate_global_indices):
        if not gidx:
            continue
        # Gate starts at the max of all its qubit availability times
        start = max(qubit_time[q] for q in gidx if q < n)
        end = start + 1
        for q in gidx:
            if q < n:
                qubit_time[q] = end
        depths.append(end)

    total_depth = max(qubit_time) if qubit_time else 0

    return {
        "circuit_depth": float(total_depth),
        "depth_per_qubit": float(total_depth) / max(n, 1),
        "gate_density": len(circuit.gates) / max(n * total_depth, 1),
    }


def compute_entanglement_layer_features(circuit: ParsedCircuit) -> Dict[str, float]:
    """
    Count the number of 'entangling layers' — groups of two-qubit gates
    separated by single-qubit gate layers. More entangling layers means
    the simulator must perform more SVD truncations.
    """
    n_entangling_gates = 0
    entangling_layers = 0
    prev_was_entangling = False

    for gate in circuit.gates:
        is_ent = gate.name in TWO_QUBIT_GATES or gate.name in THREE_QUBIT_GATES
        if is_ent:
            n_entangling_gates += 1
            if not prev_was_entangling:
                entangling_layers += 1
            prev_was_entangling = True
        else:
            prev_was_entangling = False

    return {
        "n_entangling_gates": float(n_entangling_gates),
        "n_entangling_layers": float(entangling_layers),
        "entangling_gate_ratio": n_entangling_gates / max(len(circuit.gates), 1),
    }
