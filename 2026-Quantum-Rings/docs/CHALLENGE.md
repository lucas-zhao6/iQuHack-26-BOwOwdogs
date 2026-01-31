# Circuit Fingerprint Challenge

This is an **offline modeling challenge**: you do *not* run the simulator during the hackathon.
Instead, you learn from a labeled dataset and make predictions on an unlabeled holdout set.  Holdout circuits are not distributed; organizers run your submitted code on the hidden holdout circuits to generate the scored predictions.

> You will submit code (not final predictions). Organizers will run your code on hidden holdout circuits.
> See `docs/SUBMISSION.md` for the required `predict.py` interface and submission packaging (ZIP recommended).

---

## The idea in plain language

Tensor‑network simulators are fast *when the circuit stays “compressible.”*
A standard representation is a **Matrix Product State (MPS)**, where the “hard part” is how much
entanglement you need to represent across cuts of the qubit chain.

When the simulator applies gates, it often has to **factorize and truncate** intermediate tensors
(e.g., via SVD). That truncation is controlled by a knob we call **threshold**:

- **Lower threshold** → more aggressive truncation  
  ✅ faster / less memory  
  ❌ lower fidelity

- **Higher threshold** → keep more information  
  ✅ higher fidelity  
  ❌ slower / more memory

If you want an intuition: it’s like choosing a compression quality setting. The right setting depends
heavily on **circuit structure** (entanglement patterns, depth, connectivity, etc.).

---

## Mirror circuits and “fidelity”

We can’t compute exact state fidelity for large circuits cheaply, so the dataset uses a **mirror**
benchmark:

> run `U · U†` and measure whether you return to the starting state

In an ideal simulation, a perfect mirror returns to `|0…0⟩`.
The harness turns the observed counts into a **fidelity‑like metric**.

That is why the dataset contains:

- **mirror sweeps** across thresholds (cheap-ish, used to pick the threshold)
- **one forward run** per configuration at the selected threshold (expensive, used for runtime labels)

---

## How you're scored

Conceptually, you are solving:

> **Minimize runtime, subject to mirror fidelity ≥ 0.99.**

In the real world, you would pick (CPU/GPU, precision, threshold) to satisfy the fidelity constraint while minimizing wall time.

**In this hackathon, scoring is offline and accuracy-based:** organizers compare your predicted minimum threshold rung and forward runtime against a private truth file derived from real runs.

Scoring notes:
- Your `predicted_threshold_min` must be a **ladder rung** (1, 2, 4, 8, 16, 32, 64, 128, 256).
- If you predict a rung **below** the true minimum rung, that task scores **0** (it would fail the fidelity constraint).
- Predicting a rung **above** the true minimum is penalized by rung distance.
- Runtime accuracy is scored symmetrically (over/under both count) and normalized.
- **Final ranking combines the grader score (75%) with presentation (25%).**

Full details are in `docs/SUBMISSION.md`.

---

## Practical feature engineering ideas

You have raw **OpenQASM 2.0**. You don’t need a heavyweight parser to get useful signals.

### Quick “grep‑level” signals (low effort, surprisingly strong)

From the repo root:

```powershell
# total lines (very rough proxy for size)
(Get-Content .\circuits\foo.qasm).Length

# count common 2Q gates (if your circuits use cx/cz)
(Select-String -Path .\circuits\foo.qasm -Pattern "\bcx\b" -AllMatches).Matches.Count
(Select-String -Path .\circuits\foo.qasm -Pattern "\bcz\b" -AllMatches).Matches.Count
```

Notes:
- Some files may declare custom gates (`gate ... { ... }`). A naïve `cx` count may miss gates that expand to `cx`.
- You do **not** need to fully expand custom gate definitions to be competitive.  
- Simple, approximate signals are sufficient for this challenge.
- Still, even simple token counts often correlate with runtime.

### Lightweight Python feature extraction

```python
import re
from pathlib import Path

text = Path("circuits/foo.qasm").read_text(encoding="utf-8")

n_lines = sum(1 for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("//"))
n_meas  = len(re.findall(r"\bmeasure\b", text))
n_cx    = len(re.findall(r"\bcx\b", text))
n_cz    = len(re.findall(r"\bcz\b", text))
n_1q    = len(re.findall(r"\b(h|x|y|z|s|sdg|t|tdg|rx|ry|rz|u1|u2|u3)\b", text))

print(n_lines, n_meas, n_cx, n_cz, n_1q)
```

### Structure signals (worth the extra effort)

These tend to matter a lot for tensor‑networks:

- **2‑qubit interaction graph**
  - number of edges (unique pairs touched)
  - degree stats (max degree, degree entropy)
  - clustering, connected components
- **Depth proxies**
  - even a crude “layering” heuristic helps (e.g., count of moments)
- **Entanglement / cut pressure proxies**
  - how often gates cross a chosen 1D ordering cut
  - average “span” distance between qubits in 2Q gates
- **Algorithm family fingerprints**
  - QFT‑like patterns, QNN layers, graph state patterns, arithmetic (Shor), etc.

You do *not* need perfect physics here — you’re building an accurate predictor.

---

## Modeling tips

- Predict **log(runtime)**, then exponentiate.
- Build **separate models** for CPU vs GPU and for single vs double precision, or include them as categorical features.
- A strong baseline is:
  - `predicted_threshold_min` from a classifier over the ladder
  - `predicted_forward_wall_s` from a regression model conditioned on predicted threshold and circuit features

---

## Where to look next

- `docs/DATA.md` — exact JSON schema and field meanings
- `docs/SUBMISSION.md` — canonical prediction format + validation instructions
