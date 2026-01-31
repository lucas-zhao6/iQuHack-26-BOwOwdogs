"""
Connectivity graph analysis and circuit family fingerprinting.

Builds a qubit interaction graph from two-qubit gates and extracts
graph-theoretic features. Also computes structural fingerprints for
circuit family classification.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Set
from collections import Counter

import numpy as np
import networkx as nx

from .qasm_parser import (
    ParsedCircuit, TWO_QUBIT_GATES, THREE_QUBIT_GATES, PARAMETERIZED_GATES,
)


def build_interaction_graph(circuit: ParsedCircuit) -> nx.Graph:
    """Build a weighted interaction graph where edges = two-qubit gate interactions."""
    G = nx.Graph()
    G.add_nodes_from(range(circuit.total_qubits))

    for gate, gidx in zip(circuit.gates, circuit.gate_global_indices):
        if gate.name in TWO_QUBIT_GATES and len(gidx) >= 2:
            q0, q1 = gidx[0], gidx[1]
            if G.has_edge(q0, q1):
                G[q0][q1]["weight"] += 1
            else:
                G.add_edge(q0, q1, weight=1)
        elif gate.name in THREE_QUBIT_GATES and len(gidx) >= 2:
            for i in range(len(gidx)):
                for j in range(i + 1, len(gidx)):
                    q0, q1 = gidx[i], gidx[j]
                    if G.has_edge(q0, q1):
                        G[q0][q1]["weight"] += 1
                    else:
                        G.add_edge(q0, q1, weight=1)
    return G


def compute_graph_features(circuit: ParsedCircuit) -> Dict[str, float]:
    """
    Extract graph-theoretic features from the qubit interaction graph.

    These features capture the connectivity pattern which directly affects
    MPS simulation difficulty (bond dimension growth).
    """
    G = build_interaction_graph(circuit)
    n = circuit.total_qubits
    n_edges = G.number_of_edges()
    max_possible_edges = n * (n - 1) / 2 if n > 1 else 1

    # Degree statistics
    degrees = [d for _, d in G.degree()]
    deg_arr = np.array(degrees, dtype=np.float64) if degrees else np.array([0.0])

    # Weighted degree (total gate interactions per qubit)
    w_degrees = [d for _, d in G.degree(weight="weight")]
    w_deg_arr = np.array(w_degrees, dtype=np.float64) if w_degrees else np.array([0.0])

    # Connected components
    components = list(nx.connected_components(G))
    n_components = len(components)
    largest_component = max(len(c) for c in components) if components else 0

    # Clustering coefficient
    try:
        avg_clustering = nx.average_clustering(G)
    except Exception:
        avg_clustering = 0.0

    # Degree entropy (uniformity of connectivity)
    if deg_arr.sum() > 0:
        p = deg_arr / deg_arr.sum()
        p = p[p > 0]
        degree_entropy = float(-np.sum(p * np.log2(p)))
    else:
        degree_entropy = 0.0

    # Connectivity pattern classification
    is_nn = _check_nearest_neighbor(G, n)
    is_star = _check_star(G, n)
    is_all_to_all = n_edges >= 0.8 * max_possible_edges if n > 2 else False

    # Edge weight statistics
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    w_arr = np.array(weights, dtype=np.float64) if weights else np.array([0.0])

    return {
        # Basic graph stats
        "graph_n_edges": float(n_edges),
        "graph_density": n_edges / max(max_possible_edges, 1),
        "graph_n_components": float(n_components),
        "graph_largest_component_frac": largest_component / max(n, 1),
        "graph_avg_clustering": avg_clustering,

        # Degree stats
        "graph_max_degree": float(deg_arr.max()),
        "graph_mean_degree": float(deg_arr.mean()),
        "graph_std_degree": float(deg_arr.std()),
        "graph_degree_entropy": degree_entropy,

        # Weighted degree stats
        "graph_max_wdegree": float(w_deg_arr.max()),
        "graph_mean_wdegree": float(w_deg_arr.mean()),

        # Edge weight stats
        "graph_max_edge_weight": float(w_arr.max()),
        "graph_mean_edge_weight": float(w_arr.mean()),

        # Topology classification (binary)
        "is_nearest_neighbor": float(is_nn),
        "is_star_topology": float(is_star),
        "is_all_to_all": float(is_all_to_all),
    }


def _check_nearest_neighbor(G: nx.Graph, n: int) -> bool:
    """Check if all edges are between adjacent qubits (|i-j| == 1)."""
    for u, v in G.edges():
        if abs(u - v) != 1:
            return False
    return G.number_of_edges() > 0


def _check_star(G: nx.Graph, n: int) -> bool:
    """Check if graph has star topology (one hub connecting to most others)."""
    if n < 3 or G.number_of_edges() == 0:
        return False
    degrees = dict(G.degree())
    max_deg = max(degrees.values())
    # Star if one node connects to >= 80% of others
    return max_deg >= 0.8 * (n - 1)


def compute_gate_type_fingerprint(circuit: ParsedCircuit) -> Dict[str, float]:
    """
    Compute a gate-type fingerprint vector.

    This captures the 'signature' of different circuit families:
    - QFT: heavy on cp + swap
    - QAOA: heavy on rzz + rx
    - Grover: uses ccx/rccx
    - VQE: ry + cx
    etc.
    """
    gate_counter = Counter(g.name for g in circuit.gates)
    total = max(len(circuit.gates), 1)

    # Key gate types (normalized counts)
    key_gates = [
        "cx", "cz", "cp", "rzz", "rxx", "swap", "ccx", "rccx", "cswap",
        "h", "x", "y", "z", "rx", "ry", "rz", "p",
        "u", "u1", "u2", "u3",
        "s", "sdg", "t", "tdg", "sx", "sxdg",
        "cu", "cu1", "cu3", "cry", "crx", "crz", "ch",
    ]

    features = {}
    for g in key_gates:
        features[f"gate_{g}_count"] = float(gate_counter.get(g, 0))
        features[f"gate_{g}_frac"] = gate_counter.get(g, 0) / total

    # Aggregate features
    n_1q = sum(
        gate_counter.get(g, 0)
        for g in gate_counter
        if g not in TWO_QUBIT_GATES and g not in THREE_QUBIT_GATES
    )
    n_2q = sum(gate_counter.get(g, 0) for g in gate_counter if g in TWO_QUBIT_GATES)
    n_3q = sum(gate_counter.get(g, 0) for g in gate_counter if g in THREE_QUBIT_GATES)
    n_param = sum(gate_counter.get(g, 0) for g in gate_counter if g in PARAMETERIZED_GATES)

    features["n_1q_gates"] = float(n_1q)
    features["n_2q_gates"] = float(n_2q)
    features["n_3q_gates"] = float(n_3q)
    features["n_parameterized_gates"] = float(n_param)
    features["frac_2q"] = n_2q / total
    features["frac_3q"] = n_3q / total
    features["frac_parameterized"] = n_param / total
    features["n_distinct_gate_types"] = float(len(gate_counter))
    features["total_gates"] = float(total)

    return features


def compute_param_stats(circuit: ParsedCircuit) -> Dict[str, float]:
    """
    Statistics over gate rotation angles.

    Captures the distribution of parameterized gate angles, which differ
    across circuit families (e.g., QFT has geometrically decaying angles).
    """
    all_params = []
    for gate in circuit.gates:
        for p in gate.params:
            if p is not None and math.isfinite(p):
                all_params.append(abs(p))

    if not all_params:
        return {
            "param_mean": 0.0,
            "param_std": 0.0,
            "param_min": 0.0,
            "param_max": 0.0,
            "param_n_unique": 0.0,
            "param_entropy": 0.0,
        }

    arr = np.array(all_params)
    n_unique = len(set(round(p, 8) for p in all_params))

    # Entropy of binned parameter distribution
    hist, _ = np.histogram(arr, bins=min(20, max(n_unique, 2)))
    hist = hist[hist > 0].astype(float)
    if hist.sum() > 0:
        p_hist = hist / hist.sum()
        entropy = float(-np.sum(p_hist * np.log2(p_hist)))
    else:
        entropy = 0.0

    return {
        "param_mean": float(arr.mean()),
        "param_std": float(arr.std()),
        "param_min": float(arr.min()),
        "param_max": float(arr.max()),
        "param_n_unique": float(n_unique),
        "param_entropy": entropy,
    }
