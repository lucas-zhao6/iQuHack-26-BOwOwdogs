from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple


QREG_RE = re.compile(r"^qreg\s+(\w+)\[(\d+)\]\s*;")
CREG_RE = re.compile(r"^creg\s+(\w+)\[(\d+)\]\s*;")
COMMENT_RE = re.compile(r"//.*$")
QUBIT_RE = re.compile(r"(\w+)\[(\d+)\]")


@dataclass(frozen=True)
class Qubit:
    reg: str
    idx: int


def _strip_comments(line: str) -> str:
    return COMMENT_RE.sub("", line).strip()


def _is_header(line: str) -> bool:
    return line.startswith("OPENQASM") or line.startswith("include ")


def _is_ignorable(line: str) -> bool:
    return not line or line.startswith("barrier ") or line.startswith("measure ") or line.startswith("reset ")


def _parse_qregs(lines: Iterable[str]) -> Dict[str, int]:
    qregs: Dict[str, int] = {}
    for raw in lines:
        line = _strip_comments(raw)
        if not line:
            continue
        m = QREG_RE.match(line)
        if m:
            qregs[m.group(1)] = int(m.group(2))
        if CREG_RE.match(line):
            continue
    return qregs


def _build_qubit_index(qregs: Dict[str, int]) -> Dict[Qubit, int]:
    index: Dict[Qubit, int] = {}
    offset = 0
    for reg, size in qregs.items():
        for i in range(size):
            index[Qubit(reg, i)] = offset
            offset += 1
    return index


def _extract_operands(line: str) -> List[Qubit]:
    qubits: List[Qubit] = []
    for reg, idx in QUBIT_RE.findall(line):
        qubits.append(Qubit(reg, int(idx)))
    return qubits


def parse_qasm_features(path: str | Path) -> Dict[str, int]:
    """
    Parse a QASM file and return basic circuit features:
    - one_qubit_gate_count
    - two_qubit_gate_count
    - depth (all gates)
    - two_qubit_depth (layers containing any 2-qubit gate)
    """
    text = Path(path).read_text(encoding="utf-8")
    raw_lines = text.splitlines()

    qregs = _parse_qregs(raw_lines)
    qubit_index = _build_qubit_index(qregs)
    if not qubit_index:
        return {
            "one_qubit_gate_count": 0,
            "two_qubit_gate_count": 0,
            "depth": 0,
            "two_qubit_depth": 0,
        }

    per_qubit_depth = [0] * len(qubit_index)
    one_q = 0
    two_q = 0
    two_q_layers = set()

    for raw in raw_lines:
        line = _strip_comments(raw)
        if not line or _is_header(line) or _is_ignorable(line):
            continue
        if QREG_RE.match(line) or CREG_RE.match(line):
            continue

        operands = _extract_operands(line)
        if not operands:
            continue
        indices = [qubit_index[q] for q in operands if q in qubit_index]
        if not indices:
            continue

        arity = len(indices)
        if arity == 1:
            one_q += 1
        else:
            two_q += 1

        # Schedule at the earliest available layer across involved qubits.
        layer = max(per_qubit_depth[i] for i in indices) + 1
        for i in indices:
            per_qubit_depth[i] = layer
        if arity >= 2:
            two_q_layers.add(layer)

    depth = max(per_qubit_depth) if per_qubit_depth else 0
    return {
        "one_qubit_gate_count": one_q,
        "two_qubit_gate_count": two_q,
        "depth": depth,
        "two_qubit_depth": len(two_q_layers),
    }
