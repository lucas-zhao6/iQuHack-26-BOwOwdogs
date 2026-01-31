"""
QASM 2.0 parser for feature extraction.

Parses OpenQASM 2.0 files and extracts gate-level information
needed for downstream feature engineering.
"""

from __future__ import annotations

import re
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


# Gate classification
TWO_QUBIT_GATES = frozenset([
    "cx", "cz", "cp", "rzz", "rxx", "ryy", "swap", "cu1", "cu3", "cu",
    "cry", "crx", "crz", "ch", "cy", "csx",
])
THREE_QUBIT_GATES = frozenset(["ccx", "rccx", "cswap", "ccz"])
PARAMETERIZED_GATES = frozenset([
    "rx", "ry", "rz", "p", "cp", "rzz", "rxx", "ryy",
    "u", "u1", "u2", "u3", "cu1", "cu3", "cry", "crx", "crz",
])

# Regex patterns
RE_QREG = re.compile(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]")
RE_CREG = re.compile(r"creg\s+(\w+)\s*\[\s*(\d+)\s*\]")
RE_GATE = re.compile(
    r"^(\w+)"               # gate name
    r"(?:\(([^)]*)\))?"     # optional parameters in parens
    r"\s+"                  # whitespace
    r"(.+?)\s*;$"           # qubit arguments before semicolon
)
RE_QUBIT = re.compile(r"(\w+)\s*\[\s*(\d+)\s*\]")


def _eval_param(expr: str) -> Optional[float]:
    """Safely evaluate a QASM parameter expression like 'pi/4' or '3*pi/8'."""
    expr = expr.strip()
    if not expr:
        return None
    try:
        expr_py = expr.replace("pi", str(math.pi))
        return float(eval(expr_py, {"__builtins__": {}}, {}))
    except Exception:
        return None


@dataclass
class GateInfo:
    name: str
    qubits: List[Tuple[str, int]]  # [(register_name, index), ...]
    params: List[Optional[float]] = field(default_factory=list)

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    @property
    def global_indices(self) -> List[int]:
        """Return qubits as global indices (set by parser after register mapping)."""
        return self._global_indices

    @global_indices.setter
    def global_indices(self, val: List[int]):
        self._global_indices = val


@dataclass
class ParsedCircuit:
    """Full parsed representation of an OpenQASM 2.0 circuit."""
    filename: str
    qregs: Dict[str, int]            # register_name -> size
    cregs: Dict[str, int]
    gates: List[GateInfo]
    total_qubits: int
    total_cbits: int
    n_qregs: int
    n_cregs: int

    # Derived after parsing
    gate_global_indices: List[List[int]] = field(default_factory=list)


def parse_qasm(filepath: str | Path) -> ParsedCircuit:
    """Parse an OpenQASM 2.0 file into structured gate information."""
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8")
    lines = text.splitlines()

    qregs: Dict[str, int] = {}
    cregs: Dict[str, int] = {}
    gates: List[GateInfo] = []

    # Track register ordering for global index mapping
    qreg_order: List[str] = []

    for line in lines:
        line = line.strip()

        # Skip empty, comments, headers
        if not line or line.startswith("//") or line.startswith("OPENQASM") or line.startswith("include"):
            continue

        # Parse qreg
        m = RE_QREG.match(line)
        if m:
            name, size = m.group(1), int(m.group(2))
            qregs[name] = size
            qreg_order.append(name)
            continue

        # Parse creg
        m = RE_CREG.match(line)
        if m:
            cregs[m.group(1)] = int(m.group(2))
            continue

        # Skip barrier and measure
        if line.startswith("barrier") or line.startswith("measure"):
            continue

        # Parse gate
        m = RE_GATE.match(line)
        if m:
            gate_name = m.group(1)
            param_str = m.group(2)
            qubit_str = m.group(3)

            # Parse parameters
            params = []
            if param_str:
                for p in param_str.split(","):
                    params.append(_eval_param(p))

            # Parse qubit arguments
            qubits = []
            for qm in RE_QUBIT.finditer(qubit_str):
                qubits.append((qm.group(1), int(qm.group(2))))

            if qubits:
                gates.append(GateInfo(name=gate_name, qubits=qubits, params=params))

    # Build global index mapping: register offsets
    total_qubits = sum(qregs.values())
    total_cbits = sum(cregs.values())

    reg_offset: Dict[str, int] = {}
    offset = 0
    for rname in qreg_order:
        reg_offset[rname] = offset
        offset += qregs[rname]

    # Assign global indices to each gate
    gate_global_indices = []
    for g in gates:
        gidx = []
        for rname, ridx in g.qubits:
            if rname in reg_offset:
                gidx.append(reg_offset[rname] + ridx)
            else:
                gidx.append(ridx)  # fallback
        gate_global_indices.append(gidx)
        g._global_indices = gidx

    return ParsedCircuit(
        filename=filepath.name,
        qregs=qregs,
        cregs=cregs,
        gates=gates,
        total_qubits=total_qubits,
        total_cbits=total_cbits,
        n_qregs=len(qregs),
        n_cregs=len(cregs),
        gate_global_indices=gate_global_indices,
    )
