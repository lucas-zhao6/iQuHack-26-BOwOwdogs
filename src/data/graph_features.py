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
from networkx.algorithms import approximation as nx_approx

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


def compute_graph_features(
    circuit: ParsedCircuit,
    include_expansion_structure: bool = True,
) -> Dict[str, float]:
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
    deg_min = float(deg_arr.min())
    deg_med = float(np.median(deg_arr))
    deg_p25 = float(np.percentile(deg_arr, 25))
    deg_p75 = float(np.percentile(deg_arr, 75))

    # Weighted degree (total gate interactions per qubit)
    w_degrees = [d for _, d in G.degree(weight="weight")]
    w_deg_arr = np.array(w_degrees, dtype=np.float64) if w_degrees else np.array([0.0])
    w_deg_min = float(w_deg_arr.min())
    w_deg_med = float(np.median(w_deg_arr))
    w_deg_p25 = float(np.percentile(w_deg_arr, 25))
    w_deg_p75 = float(np.percentile(w_deg_arr, 75))
    w_deg_std = float(w_deg_arr.std())

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

    # Degree-based width proxies
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        try:
            core_numbers = nx.core_number(G)
            degeneracy = float(max(core_numbers.values())) if core_numbers else 0.0
        except Exception:
            degeneracy = 0.0
        try:
            arboricity = float(nx_approx.arboricity(G))
        except Exception:
            arboricity = 0.0
        try:
            tw_md, _ = nx_approx.treewidth_min_degree(G)
            treewidth_min_degree = float(tw_md)
        except Exception:
            treewidth_min_degree = 0.0
        try:
            tw_mf, _ = nx_approx.treewidth_min_fill_in(G)
            treewidth_min_fill = float(tw_mf)
        except Exception:
            treewidth_min_fill = 0.0
    else:
        degeneracy = 0.0
        arboricity = 0.0
        treewidth_min_degree = 0.0
        treewidth_min_fill = 0.0

    # Pathwidth surrogates (ordering-based width proxies)
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        natural_order = list(range(n))
        heuristic_order = _min_degree_order(G)
        bw_nat = _bandwidth(G, natural_order)
        bw_heur = _bandwidth(G, heuristic_order)
        cw_nat = _cutwidth(G, natural_order)
        cw_heur = _cutwidth(G, heuristic_order)
        la_nat = _linear_arrangement_cost(G, natural_order)
        la_heur = _linear_arrangement_cost(G, heuristic_order)
        bw_min = min(bw_nat, bw_heur)
        cw_min = min(cw_nat, cw_heur)
        la_min = min(la_nat, la_heur)
    else:
        bw_nat = bw_heur = bw_min = 0.0
        cw_nat = cw_heur = cw_min = 0.0
        la_nat = la_heur = la_min = 0.0

    # Separator metrics (heuristics)
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        min_bal_sep = _balanced_separator_size(G)
        rb_max_cut, rb_avg_cut = _recursive_bisection_cuts(G)
    else:
        min_bal_sep = 0.0
        rb_max_cut = 0.0
        rb_avg_cut = 0.0

    # Expansion/structure metrics
    if include_expansion_structure:
        if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
            try:
                largest_nodes = max(components, key=len) if components else set()
                H = G.subgraph(largest_nodes)
                if H.number_of_nodes() > 1 and nx.is_connected(H):
                    avg_shortest_path = float(nx.average_shortest_path_length(H))
                    diameter = float(nx.diameter(H))
                else:
                    avg_shortest_path = 0.0
                    diameter = 0.0
            except Exception:
                avg_shortest_path = 0.0
                diameter = 0.0
            conductance = _best_cut_conductance(G)
        else:
            avg_shortest_path = 0.0
            diameter = 0.0
            conductance = 0.0

        sparsity = 1.0 - (n_edges / max(max_possible_edges, 1))
        clustering_over_density = avg_clustering / (n_edges / max(max_possible_edges, 1) + 1e-12)
    else:
        avg_shortest_path = 0.0
        diameter = 0.0
        conductance = 0.0
        sparsity = 0.0
        clustering_over_density = 0.0

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
        "graph_avg_shortest_path": avg_shortest_path,
        "graph_diameter": diameter,
        "graph_conductance_best_cut": conductance,
        "graph_sparsity": sparsity,
        "graph_clustering_over_density": clustering_over_density,

        # Degree stats
        "graph_max_degree": float(deg_arr.max()),
        "graph_mean_degree": float(deg_arr.mean()),
        "graph_std_degree": float(deg_arr.std()),
        "graph_min_degree": deg_min,
        "graph_median_degree": deg_med,
        "graph_p25_degree": deg_p25,
        "graph_p75_degree": deg_p75,
        "graph_degree_entropy": degree_entropy,
        "graph_degeneracy": degeneracy,
        "graph_arboricity": arboricity,
        "graph_treewidth_min_degree": treewidth_min_degree,
        "graph_treewidth_min_fill": treewidth_min_fill,

        # Pathwidth surrogates
        "graph_bandwidth_natural": bw_nat,
        "graph_bandwidth_heuristic": bw_heur,
        "graph_bandwidth_min": bw_min,
        "graph_cutwidth_natural": cw_nat,
        "graph_cutwidth_heuristic": cw_heur,
        "graph_cutwidth_min": cw_min,
        "graph_linear_arrangement_natural": la_nat,
        "graph_linear_arrangement_heuristic": la_heur,
        "graph_linear_arrangement_min": la_min,

        # Separator metrics (heuristics)
        "graph_min_balanced_separator": min_bal_sep,
        "graph_recursive_bisection_max_cut": rb_max_cut,
        "graph_recursive_bisection_avg_cut": rb_avg_cut,

        # Weighted degree stats
        "graph_max_wdegree": float(w_deg_arr.max()),
        "graph_mean_wdegree": float(w_deg_arr.mean()),
        "graph_std_wdegree": w_deg_std,
        "graph_min_wdegree": w_deg_min,
        "graph_median_wdegree": w_deg_med,
        "graph_p25_wdegree": w_deg_p25,
        "graph_p75_wdegree": w_deg_p75,

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


def _min_degree_order(G: nx.Graph) -> List[int]:
    """Greedy minimum-degree elimination order."""
    H = G.copy()
    order = []
    while H.number_of_nodes() > 0:
        node = min(H.degree(), key=lambda x: (x[1], x[0]))[0]
        order.append(node)
        H.remove_node(node)
    return order


def _bandwidth(G: nx.Graph, order: List[int]) -> float:
    """Maximum edge span under the given ordering."""
    pos = {n: i for i, n in enumerate(order)}
    if G.number_of_edges() == 0:
        return 0.0
    return float(max(abs(pos[u] - pos[v]) for u, v in G.edges()))


def _cutwidth(G: nx.Graph, order: List[int]) -> float:
    """Maximum number of edges crossing any cut in the ordering."""
    pos = {n: i for i, n in enumerate(order)}
    n_nodes = len(order)
    if G.number_of_edges() == 0 or n_nodes <= 1:
        return 0.0
    max_cut = 0
    for i in range(n_nodes - 1):
        cut = 0
        for u, v in G.edges():
            pu, pv = pos[u], pos[v]
            if (pu <= i < pv) or (pv <= i < pu):
                cut += 1
        if cut > max_cut:
            max_cut = cut
    return float(max_cut)


def _linear_arrangement_cost(G: nx.Graph, order: List[int]) -> float:
    """Sum of edge lengths under the given ordering."""
    pos = {n: i for i, n in enumerate(order)}
    if G.number_of_edges() == 0:
        return 0.0
    return float(sum(abs(pos[u] - pos[v]) for u, v in G.edges()))


def _balanced_separator_size(G: nx.Graph) -> float:
    """Heuristic minimum balanced separator size via KL bisection."""
    if G.number_of_nodes() <= 1 or G.number_of_edges() == 0:
        return 0.0
    try:
        part_a, part_b = nx.algorithms.community.kernighan_lin_bisection(G)
    except Exception:
        return 0.0
    if len(part_a) == 0 or len(part_b) == 0:
        return 0.0
    part_b = set(part_b)
    boundary_a = {u for u in part_a if any(v in part_b for v in G.neighbors(u))}
    boundary_b = {v for v in part_b if any(u in part_a for u in G.neighbors(v))}
    return float(min(len(boundary_a), len(boundary_b)))


def _recursive_bisection_cuts(G: nx.Graph, max_depth: int = 10) -> Tuple[float, float]:
    """Heuristic recursive bisection using KL; returns (max_cut, avg_cut)."""
    cuts: List[int] = []

    def recurse(nodes: Set[int], depth: int) -> None:
        if depth >= max_depth:
            return
        H = G.subgraph(nodes)
        if H.number_of_nodes() <= 1 or H.number_of_edges() == 0:
            return
        try:
            part_a, part_b = nx.algorithms.community.kernighan_lin_bisection(H)
        except Exception:
            return
        if len(part_a) == 0 or len(part_b) == 0:
            return
        part_b_set = set(part_b)
        cut_edges = 0
        for u in part_a:
            for v in H.neighbors(u):
                if v in part_b_set:
                    cut_edges += 1
        cuts.append(cut_edges)
        recurse(set(part_a), depth + 1)
        recurse(set(part_b), depth + 1)

    recurse(set(G.nodes()), 0)
    if not cuts:
        return 0.0, 0.0
    return float(max(cuts)), float(sum(cuts) / len(cuts))


def _best_cut_conductance(G: nx.Graph) -> float:
    """Heuristic conductance of a balanced cut using KL bisection."""
    if G.number_of_nodes() <= 1 or G.number_of_edges() == 0:
        return 0.0
    try:
        part_a, part_b = nx.algorithms.community.kernighan_lin_bisection(G)
    except Exception:
        return 0.0
    if len(part_a) == 0 or len(part_b) == 0:
        return 0.0
    part_a = set(part_a)
    part_b = set(part_b)
    cut_edges = 0
    vol_a = 0
    vol_b = 0
    for u in G.nodes():
        deg = G.degree(u)
        if u in part_a:
            vol_a += deg
        else:
            vol_b += deg
    for u, v in G.edges():
        if (u in part_a and v in part_b) or (u in part_b and v in part_a):
            cut_edges += 1
    denom = min(vol_a, vol_b)
    if denom <= 0:
        return 0.0
    return float(cut_edges / denom)


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
