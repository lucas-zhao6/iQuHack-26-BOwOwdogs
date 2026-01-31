# Data format (public)

The public dataset is:

- `data/hackathon_public.json`

This file is provided for **training and validation only**.  Final scoring is performed by organizers by running submitted code against hidden holdout circuits; data/holdout_public.json contains the public task definitions (IDs + CPU/GPU + precision) but not circuit files or labels.

---

## Identifiers

### Circuit ID
A circuit is identified by its **OpenQASM filename**:

- `file` (example: `qft_indep_qiskit_30.qasm`)

Use this as the join key between:
- `circuits[]` entries
- `results[]` entries
- and the actual file in `circuits/`

---

## Backends and precision

### Backend
We expose two backend labels:

- `CPU`
- `GPU`

### Precision
- `single`
- `double`

---

## High-level structure

`hackathon_public.json` contains two top-level arrays:

- `circuits`: circuit identity + family + source
- `results`: one record per `(file, backend, precision)`

---

## Note on the holdout dataset

The unlabeled holdout file (`data/holdout_public.json`) uses a **task-oriented schema**
rather than the `circuits[]` / `results[]` structure described here.

Each holdout task corresponds to a single:
- circuit
- processor (`CPU` or `GPU`)
- precision (`single` or `double`)

and is identified by a unique `id`.

Holdout tasks do not include circuit filenames or QASM.  At scoring time, organizers provide a private ID map that resolves each holdout task `id` to the corresponding hidden QASM file.

See `docs/SUBMISSION.md` for the holdout task format and submission requirements.

---

## `circuits[]` glossary

Each entry includes:

- `file`: QASM filename (the circuit ID)
- `family`: a coarse family label (e.g., `QFT`, `GHZ`, `TwoLocalRandom`)
- `source`: `{name, url, notes}` describing where the circuit came from
- `n_qubits`: declared qubit count (from the circuit header / harness features)

---

## `results[]` glossary

Each entry corresponds to one simulator configuration:

- `file`: QASM filename (joins to `circuits[]`)
- `backend`: `CPU` or `GPU`
- `precision`: `single` or `double`
- `status`: overall status (e.g., `ok`, `no_threshold_met`, `runner_failed`)

And includes (when available):

- `selection`: how the harness chose the “production” threshold for forward runs
- `threshold_sweep`: mirror runs over a threshold ladder
- `verify` (optional): extra mirror verification (`p_return_zero`)
- `state_setup` (optional): a 1-shot forward run at the selected threshold
- `forward` (optional): a 10,000-shot forward run at the selected threshold (Top-K included)
- `forward_timing_estimates` (optional): convenience estimates derived from `state_setup` + `forward`

Modeling note: Entries with status != "ok" indicate runs that did not successfully produce a usable threshold or forward result (e.g., timeouts or failures).
These rows may be excluded from training or handled separately; they will not appear in the holdout set.

---

## Mirror runs and threshold sweeps

### What is a mirror run?
A mirror circuit is:

> `U` followed by `U†`

The ideal output is the **all-zeros** bitstring.

The threshold sweep runs the **mirror circuit** at multiple thresholds to measure how fidelity changes with the simulator’s truncation parameter.

### `threshold_sweep[]` fields
Each element includes:

- `threshold`: truncation / bond-dimension control knob (ladder rung)
- `shots`: number of shots used for the mirror estimate (typically 200)
- `sdk_get_fidelity`: SDK-reported mirror fidelity signal
- `p_return_zero`: counts-based estimate of returning to all-zeros (probability)
- `run_wall_s`: wall time for the mirror run at this threshold
- `peak_rss_mb`: memory usage (RSS)
- `note`: memory fields are provided for exploratory analysis only and are **not used in scoring**.
- `returncode`: `0` is success; `124` typically indicates a timeout
- `note`: status text (e.g., `ok`, `mirror_timeout`)

**Note:** In the `qasm_counts` mirror mode used for this dataset, `sdk_get_fidelity` closely tracks `p_return_zero`. We include both for transparency.

---

## Threshold selection

### `selection` fields
- `target`: the target mirror fidelity used for selection (e.g., 0.99)
- `selected_threshold`: the threshold chosen for forward runs
- `selected_mirror_metric_value`: mirror fidelity at the selected threshold
- `stop_when`: selection strategy (typically “first_cross”)

Derived label (training)
For training purposes, teams typically derive the true minimum threshold by scanning threshold_sweep in increasing order and selecting the first rung where the mirror fidelity meets the target (≥ 0.99).
This corresponds to the organizer-side field true_threshold_min, which is not included in the public dataset.

If no rung reaches the target by the maximum threshold, the entry is labeled `no_threshold_met`, and the selected threshold is typically the largest rung attempted.

---

## Forward runs (distributional artifact)

For each `(file, backend, precision)`, we also run the **original circuit forward** at the selected threshold:

- `state_setup` is usually a `shots = 1` run (setup proxy)
- `forward` is usually a `shots = 10000` run

### `forward.topk`
The dataset stores a Top-K list of bitstrings and counts:

- `forward.topk_k` is the configured K (typically 50)
- `forward.topk` may contain **fewer than K entries** if the circuit produces fewer than K unique outcomes
  - Example: GHZ states often have only 2 dominant outcomes; deterministic circuits may have only 1.

---

## Timing: what you can and cannot model from public data

### Good news
You can model:
- mirror fidelity vs threshold (from `threshold_sweep`)
- mirror runtime vs threshold (from `threshold_sweep.run_wall_s`)
- total forward runtime at the selected threshold (from `forward.run_wall_s`)
- and a rough split between setup and per-shot cost at the selected threshold.

### Limitation (important)
We do **not** provide forward timing at every threshold rung.
So you generally cannot build a clean forward-runtime-vs-threshold curve from the public dataset alone.

### Workaround provided
If both `state_setup` (1 shot) and `forward` (10k shots) exist and succeeded, we include:

- `forward_timing_estimates.estimated_per_shot_s`
- `forward_timing_estimates.estimated_setup_s`

Computed as:
- `per_shot ≈ (t_forward - t_state_setup) / (shots_forward - shots_state_setup)`
- `setup ≈ t_state_setup - per_shot * shots_state_setup`

These estimates are only meaningful at the selected threshold.

---

## Practical modeling tips

- Runtime spans orders of magnitude: model `log(run_wall_s)` or use robust losses.
- Clamp tiny numeric overshoots (some fidelity values may appear slightly outside [0,1] due to floating point).
- Circuits are intentionally diverse: some are easy/structured; some are hard and may not meet the target by the maximum threshold.
- When creating train/test splits, split **by `file`** (circuit-level). Keep all `(backend, precision)` rows for a given circuit in the same split.
