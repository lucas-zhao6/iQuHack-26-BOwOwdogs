import sys
from pathlib import Path
import unittest

# Ensure src/ is on the import path when running tests from repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parse import parse_qasm_features  # noqa: E402


class TestParseShor(unittest.TestCase):
    def test_shor_15_4_indep_qiskit_18(self) -> None:
        qasm_path = ROOT / "circuits" / "shor_15_4_indep_qiskit_18.qasm"
        features = parse_qasm_features(qasm_path)
        self.assertEqual(
            features,
            {
                "one_qubit_gate_count": 1683,
                "two_qubit_gate_count": 8192,
                "depth": 6213,
                "two_qubit_depth": 5779,
            },
        )


if __name__ == "__main__":
    unittest.main()
