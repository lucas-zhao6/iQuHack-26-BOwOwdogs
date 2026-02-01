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
    n = max(circuit.total_qubits, 1)
    spans_norm = spans_arr / max(n - 1, 1)
    if spans_arr.size > 0:
        span_p25 = float(np.quantile(spans_arr, 0.25))
        span_p50 = float(np.quantile(spans_arr, 0.50))
        span_p75 = float(np.quantile(spans_arr, 0.75))
        span_p90 = float(np.quantile(spans_arr, 0.90))
        span_p25_norm = float(np.quantile(spans_norm, 0.25))
        span_p50_norm = float(np.quantile(spans_norm, 0.50))
        span_p75_norm = float(np.quantile(spans_norm, 0.75))
        span_p90_norm = float(np.quantile(spans_norm, 0.90))
    else:
        span_p25 = span_p50 = span_p75 = span_p90 = 0.0
        span_p25_norm = span_p50_norm = span_p75_norm = span_p90_norm = 0.0

    nn_count = sum(1 for s in spans if s == 1)

    return {
        "max_gate_span": float(spans_arr.max()),
        "mean_gate_span": float(spans_arr.mean()),
        "std_gate_span": float(spans_arr.std()),
        "span_p25": span_p25,
        "span_p50": span_p50,
        "span_p75": span_p75,
        "span_p90": span_p90,
        "mean_gate_span_norm": float(spans_norm.mean()),
        "std_gate_span_norm": float(spans_norm.std()),
        "span_q90_norm": float(np.quantile(spans_norm, 0.9)),
        "span_p25_norm": span_p25_norm,
        "span_p50_norm": span_p50_norm,
        "span_p75_norm": span_p75_norm,
        "span_p90_norm": span_p90_norm,
        "frac_nearest_neighbor": nn_count / len(spans),
        "frac_long_range": sum(1 for s in spans if s > circuit.total_qubits // 4) / len(spans),
        "connectivity_strain": float((spans_arr ** 2).mean()),
        "connectivity_strain_norm": float((spans_norm ** 2).mean()),
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
    layer_gate_counts: Dict[int, int] = {}
    layer_active_qubits: Dict[int, set[int]] = {}
    layer_2q_counts: Dict[int, int] = {}
    layer_3q_counts: Dict[int, int] = {}

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

        layer_idx = end - 1
        layer_gate_counts[layer_idx] = layer_gate_counts.get(layer_idx, 0) + 1
        if layer_idx not in layer_active_qubits:
            layer_active_qubits[layer_idx] = set()
        layer_active_qubits[layer_idx].update(q for q in gidx if q < n)

        if gate.name in TWO_QUBIT_GATES:
            layer_2q_counts[layer_idx] = layer_2q_counts.get(layer_idx, 0) + 1
        elif gate.name in THREE_QUBIT_GATES:
            layer_3q_counts[layer_idx] = layer_3q_counts.get(layer_idx, 0) + 1

    total_depth = max(qubit_time) if qubit_time else 0

    total_gates = len(circuit.gates)
    if total_depth > 0:
        layer_counts = np.array(
            [layer_gate_counts.get(i, 0) for i in range(total_depth)], dtype=np.float64
        )
        active_counts = np.array(
            [len(layer_active_qubits.get(i, set())) for i in range(total_depth)],
            dtype=np.float64,
        )
        layer_2q = np.array(
            [layer_2q_counts.get(i, 0) for i in range(total_depth)], dtype=np.float64
        )
        layer_3q = np.array(
            [layer_3q_counts.get(i, 0) for i in range(total_depth)], dtype=np.float64
        )
    else:
        layer_counts = np.zeros(1, dtype=np.float64)
        active_counts = np.zeros(1, dtype=np.float64)
        layer_2q = np.zeros(1, dtype=np.float64)
        layer_3q = np.zeros(1, dtype=np.float64)

    avg_parallel = total_gates / max(total_depth, 1)

    return {
        "circuit_depth": float(total_depth),
        "depth_per_qubit": float(total_depth) / max(n, 1),
        "gate_density": total_gates / max(n * total_depth, 1),
        "avg_parallel_gates": float(avg_parallel),
        "max_parallel_gates": float(layer_counts.max()),
        "std_parallel_gates": float(layer_counts.std()),
        "parallel_gates_per_qubit": float(avg_parallel) / max(n, 1),
        "max_parallel_gates_per_qubit": float(layer_counts.max()) / max(n, 1),
        "avg_active_qubits": float(active_counts.mean()),
        "max_active_qubits": float(active_counts.max()),
        "std_active_qubits": float(active_counts.std()),
        "active_qubit_ratio": float(active_counts.mean()) / max(n, 1),
        "avg_2q_gates_per_layer": float(layer_2q.mean()),
        "avg_3q_gates_per_layer": float(layer_3q.mean()),
        "max_2q_gates_per_layer": float(layer_2q.max()),
        "max_3q_gates_per_layer": float(layer_3q.max()),
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
    current_run = 0
    run_lengths = []

    for gate in circuit.gates:
        is_ent = gate.name in TWO_QUBIT_GATES or gate.name in THREE_QUBIT_GATES
        if is_ent:
            n_entangling_gates += 1
            if not prev_was_entangling:
                entangling_layers += 1
                current_run = 1
            else:
                current_run += 1
            prev_was_entangling = True
        else:
            prev_was_entangling = False
            if current_run > 0:
                run_lengths.append(current_run)
                current_run = 0

    if current_run > 0:
        run_lengths.append(current_run)

    if entangling_layers > 0:
        avg_per_layer = n_entangling_gates / entangling_layers
    else:
        avg_per_layer = 0.0

    if run_lengths:
        run_arr = np.array(run_lengths, dtype=np.float64)
        max_run = float(run_arr.max())
        mean_run = float(run_arr.mean())
        std_run = float(run_arr.std())
    else:
        max_run = 0.0
        mean_run = 0.0
        std_run = 0.0

    return {
        "n_entangling_gates": float(n_entangling_gates),
        "n_entangling_layers": float(entangling_layers),
        "entangling_gate_ratio": n_entangling_gates / max(len(circuit.gates), 1),
        "entangling_gates_per_layer": float(avg_per_layer),
        "entangling_run_max": max_run,
        "entangling_run_mean": mean_run,
        "entangling_run_std": std_run,
    }
