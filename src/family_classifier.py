"""
Circuit family classifier.

Identifies the algorithm family of a quantum circuit based on its
structural fingerprint (gate types, connectivity pattern, etc.).
This is critical for holdout circuits where filenames are anonymized.

Uses a rule-based approach (more robust than ML with only 36 training
examples per family) combined with a fallback nearest-neighbor classifier.
"""

from __future__ import annotations

from typing import Dict, Optional
from collections import Counter

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from .data.qasm_parser import ParsedCircuit, TWO_QUBIT_GATES, THREE_QUBIT_GATES


# Rule-based family signatures derived from training circuit analysis.
# Rules are evaluated in PRIORITY ORDER (first match wins).
# Each rule is a tuple: (family_name, rule_dict).
FAMILY_RULES_ORDERED = [
    # ── Highly specific families first (unique gate signatures) ──
    ("Shor", {
        "required_gates": {"cp", "cx", "h"},
        "has_any": {"cswap"},
        "check": lambda c, gc: c.n_qregs >= 3,
    }),
    ("Pricing_Call", {
        "required_gates": {"cx", "ry"},
        "has_any": {"ccx", "cry"},
    }),
    ("GraphState", {
        "required_gates": {"cz", "h"},
        "forbidden_gates": {"cx", "cp", "ry", "rx", "rzz"},
        "max_gate_types": 2,
    }),
    ("W_State", {
        "required_gates": {"ry", "cz", "cx"},
        "forbidden_gates": {"cp", "rzz", "u2", "h"},
    }),
    ("Ground_State", {
        "required_gates": {"cz", "u2"},
        "forbidden_gates": {"cx", "cp", "rzz", "ry", "ccx"},
    }),
    ("Portfolio_VQE", {
        "required_gates": {"ry", "cz"},
        "forbidden_gates": {"cx", "cp", "rzz", "u2", "u3", "ccx"},
    }),

    # ── QAOA family (rzz is unique) ──
    ("QAOA", {
        "required_gates": {"rzz", "rx", "h"},
        "forbidden_gates": {"cx", "cz", "cp", "ry", "ccx"},
    }),
    ("Portfolio_QAOA", {
        "required_gates": {"rzz", "rx"},
        "has_any": {"u2", "u3"},
        "forbidden_gates": {"cx", "cz", "cp", "ccx"},
    }),

    # ── Grover family (ccx/rccx/cu are unique) ──
    ("Grover_V_Chain", {
        "required_gates": {"cx", "cp"},
        "has_any": {"rccx", "ccx"},
        "check": lambda c, gc: c.n_qregs >= 2,
    }),
    ("Grover_NoAncilla", {
        # grover-noancilla has: cx, cp, cu, u, h, x and 2 regs (q + flag)
        "required_gates": {"cx", "cp"},
        "has_any": {"cu", "cu1"},
        "forbidden_gates": {"rzz", "cz", "ry", "cswap", "swap", "rccx"},
    }),

    # ── QNN (cx + p is unique) ──
    ("QNN", {
        "required_gates": {"cx", "p"},
        "has_any": {"u2", "ry"},
        "forbidden_gates": {"cz", "cp", "rzz", "ccx"},
    }),

    # ── cx+ry only circuits: distinguish by connectivity ──
    ("TwoLocalRandom", {
        "required_gates": {"ry", "cx"},
        "forbidden_gates": {"cz", "cp", "rzz", "u2", "u3", "ccx", "p"},
        "check": lambda c, gc: gc.get("is_all_to_all", 0) > 0.5,
    }),
    ("VQE", {
        "required_gates": {"ry", "cx"},
        "forbidden_gates": {"cz", "cp", "rzz", "u2", "u3", "ccx", "p"},
        "check": lambda c, gc: gc.get("is_nearest_neighbor", 0) > 0.5,
    }),

    # ── QPE_Exact vs QFT (both use cp+h+swap, but QPE has x and 2 regs) ──
    ("QPE_Exact", {
        "required_gates": {"cp", "h", "x"},
        "has_any": {"swap"},
        "forbidden_gates": {"rzz", "ccx", "ry", "cx", "cswap", "cu"},
        "check": lambda c, gc: c.n_qregs >= 2,
    }),
    ("QFT_Entangled", {
        "required_gates": {"cp", "h", "cx"},
        "has_any": {"swap"},
        "forbidden_gates": {"rzz", "ccx", "cswap", "ry", "x"},
    }),
    ("QFT", {
        "required_gates": {"cp", "h"},
        "has_any": {"swap"},
        "forbidden_gates": {"cx", "rzz", "ccx", "cswap", "x"},
    }),

    # ── AE (cp + cx + multi-register) ──
    ("Amplitude_Estimation", {
        "required_gates": {"cp", "cx"},
        "has_any": {"u", "u3", "u2"},
        "check": lambda c, gc: c.n_qregs >= 2,
    }),

    # ── Deutsch-Jozsa (star topology) ──
    ("Deutsch_Jozsa", {
        "required_gates": {"cx"},
        "has_any": {"u2", "h"},
        "check": lambda c, gc: gc.get("is_star_topology", 0) > 0.5,
    }),

    # ── CutBell vs GHZ: both cx+h only, distinguished by span ──
    ("CutBell", {
        "required_gates": {"cx", "h"},
        "forbidden_gates": {"cz", "cp", "ry", "rz", "rx", "rzz", "u2", "u3"},
        "max_gate_types": 2,
        "check": lambda c, gc: _has_long_range_only(c),
    }),
    ("GHZ", {
        "required_gates": {"cx", "h"},
        "forbidden_gates": {"cz", "cp", "ry", "rz", "rx", "rzz", "u2", "u3"},
        "max_gate_types": 2,
        "check": lambda c, gc: gc.get("is_nearest_neighbor", 0) > 0.5,
    }),
]


def _has_long_range_only(circuit: ParsedCircuit) -> bool:
    """Check if all two-qubit gates span more than half the qubits."""
    n = circuit.total_qubits
    for gate, gidx in zip(circuit.gates, circuit.gate_global_indices):
        if gate.name in TWO_QUBIT_GATES and len(gidx) >= 2:
            span = abs(gidx[0] - gidx[1])
            if span < n // 3:
                return False
    return True


def classify_family_rules(
    circuit: ParsedCircuit,
    graph_features: Dict[str, float],
) -> Optional[str]:
    """
    Classify a circuit's family using priority-ordered rule matching.

    Rules are evaluated in order; the first rule that fully matches wins.
    This avoids ambiguity when multiple rules could match with similar scores.

    Returns the family name or None if no rule matches.
    """
    gate_set = set(g.name for g in circuit.gates)

    for family, rules in FAMILY_RULES_ORDERED:
        # Check required gates
        required = rules.get("required_gates", set())
        if not required.issubset(gate_set):
            continue

        # Check forbidden gates
        forbidden = rules.get("forbidden_gates", set())
        if forbidden.intersection(gate_set):
            continue

        # Check has_any
        has_any = rules.get("has_any", set())
        if has_any and not has_any.intersection(gate_set):
            continue

        # Check max gate types
        max_types = rules.get("max_gate_types")
        if max_types is not None and len(gate_set) > max_types:
            continue

        # Check custom function
        check_fn = rules.get("check")
        if check_fn is not None and not check_fn(circuit, graph_features):
            continue

        return family

    return None


def classify_family_combined(
    circuit: ParsedCircuit,
    graph_features: Dict[str, float],
    gate_features: Dict[str, float],
) -> str:
    """
    Classify circuit family using rules first, then heuristic fallback.
    """
    # Try rule-based first
    result = classify_family_rules(circuit, graph_features)
    if result is not None:
        return result

    # Heuristic fallback based on dominant patterns
    gate_set = set(g.name for g in circuit.gates)

    if "rzz" in gate_set:
        return "QAOA"  # or Portfolio_QAOA
    if "cswap" in gate_set:
        return "Shor"
    if "rccx" in gate_set or "ccx" in gate_set:
        if circuit.n_qregs >= 2:
            return "Grover_V_Chain"
        return "Grover_NoAncilla"
    if "cry" in gate_set:
        return "Pricing_Call"
    if gate_features.get("is_all_to_all", 0) > 0.5:
        if "cz" in gate_set and "ry" in gate_set:
            return "Portfolio_VQE"
        if "cz" in gate_set and "u2" in gate_set:
            return "Ground_State"
        if "cx" in gate_set and "ry" in gate_set:
            return "TwoLocalRandom"
        if "cx" in gate_set and "p" in gate_set:
            return "QNN"
    if "cp" in gate_set and circuit.n_qregs >= 2:
        return "Amplitude_Estimation"
    if "cp" in gate_set and "swap" in gate_set:
        return "QFT"

    return "Unknown"


class FamilyClassifierKNN:
    """
    KNN-based family classifier trained on the 36 training circuits.
    Used as an ensemble component alongside rule-based classification.
    """

    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
        self.feature_names: list = []
        self._fitted = False

    def fit(self, feature_dicts: list[Dict[str, float]], labels: list[str]):
        """Train on feature dicts and family labels."""
        if not feature_dicts:
            return

        self.feature_names = sorted(feature_dicts[0].keys())
        X = np.array([
            [d.get(f, 0.0) for f in self.feature_names]
            for d in feature_dicts
        ])
        y = np.array(labels)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.knn.fit(X_scaled, y)
        self._fitted = True

    def predict(self, feature_dict: Dict[str, float]) -> str:
        """Predict family for a single circuit."""
        if not self._fitted:
            return "Unknown"

        x = np.array([[feature_dict.get(f, 0.0) for f in self.feature_names]])
        x_scaled = self.scaler.transform(x)
        return self.knn.predict(x_scaled)[0]

    def predict_proba(self, feature_dict: Dict[str, float]) -> Dict[str, float]:
        """Return probability distribution over families."""
        if not self._fitted:
            return {"Unknown": 1.0}

        x = np.array([[feature_dict.get(f, 0.0) for f in self.feature_names]])
        x_scaled = self.scaler.transform(x)
        probs = self.knn.predict_proba(x_scaled)[0]
        return dict(zip(self.knn.classes_, probs))
