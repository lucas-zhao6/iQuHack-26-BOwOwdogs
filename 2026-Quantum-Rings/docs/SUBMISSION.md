# Submission Instructions

This document defines the **canonical submission format and scoring workflow** for the  
**Circuit Fingerprint Challenge (iQuHACK 2026)**.

In this challenge:

- Teams submit **code**, not final scored predictions.
- **Organizers run your code** on **hidden holdout circuits** to generate the predictions that are scored.
- Any predictions JSON you upload yourself is **debug-only** and **not used for final scoring**.

---

## What you submit (required)

Submit the following items via the official Google Form.

### 1) Submission file (**exactly one file**)

You must upload **one file**:

- **Recommended:** a single **ZIP** containing `predict.py` plus any helper modules and model artifacts  
- **Allowed:** a single, fully self-contained `predict.py` (only if it has **no local imports** such as `from src ...`)

> If your code relies on helper modules or trained models, you **must** submit a ZIP.

---

### 2) Run command (required)

Provide the **exact command** organizers should run to execute your submission.

Example:
```bash
python predict.py --tasks <TASKS_JSON> --circuits <CIRCUITS_DIR> --id-map <ID_MAP> --out predictions.json
```

If your code requires additional flags (e.g. `--artifacts`), include them here.

---

### 3) Short write-up (required)

A brief description of your approach (PDF or Markdown).  
Focus on:
- features extracted
- modeling strategy
- validation approach
- known limitations

---

## Optional uploads (not required for scoring)

The following items are **optional** and do **not** affect automated scoring:

- **Presentation slides** (PDF or PPT)  
  - Used only for the in-person presentation component
- **Debug predictions files** (`predictions.json` or `predictions.normalized.json`)  
  - Helpful if organizers need to diagnose formatting issues

---

## Submission file formats

### Option A: single-file `predict.py` (simple baselines only)

Upload **only** `predict.py`.

Requirements:
- Must be completely self-contained
- Must not rely on local imports or additional files

---

### Option B: project ZIP (recommended)

Upload a single `.zip` file that unpacks into:

```
predict.py
requirements.txt        (recommended)
src/                    (optional helper modules)
artifacts/              (optional trained models)
```

Rules:
- `predict.py` **must be at the ZIP root**
- Do **not** nest everything under an extra top-level directory
- Any non-standard files your code loads **must be included in the ZIP**

---

## Required predictor interface

Organizers will run your submission using a command like:

```bash
python predict.py   --tasks <HOLDOUT_TASKS_JSON>   --circuits <HOLDOUT_QASM_DIR>   --id-map <ID_MAP_JSON>   --out predictions.json
```

Your script **must** support these arguments.

### Argument definitions

- `--tasks`  
  Path to the public holdout task list (IDs + CPU/GPU + precision)

- `--circuits`  
  Directory containing **hidden holdout QASM files** (provided by organizers)

- `--id-map`  
  JSON mapping from public task `id` to QASM filename  
  (provided by organizers at scoring time)

- `--out`  
  Output path for predictions JSON

### Additional arguments
You may define extra flags (e.g. `--artifacts`, `--seed`, `--no-conservative-bump`).  
If required, include them in your run command and use **relative paths** where possible.

---

## Holdout task file (`data/holdout_public.json`)

Each holdout task includes only:
- `id` (opaque, e.g. `H001`)
- `processor` (`CPU` or `GPU`)
- `precision` (`single` or `double`)

Holdout tasks **do not** include circuit filenames or QASM.

---

## ID map file (`--id-map`, provided by organizers)

At scoring time, organizers provide a private mapping file that resolves each task `id`
to the corresponding hidden QASM filename.

Example structure:
```json
{
  "schema": "iqhack_holdout_id_map_v1",
  "entries": [
    { "id": "H001", "qasm_file": "SOME_FILE.qasm" }
  ]
}
```

Your code should:
1. Load the mapping
2. Resolve `id → qasm_file`
3. Read QASM from `Path(--circuits) / qasm_file`

---

## Predictions output format

Your code must write **exactly one prediction per task ID**.

Required fields:

| Field | Type | Description |
|------|------|-------------|
| `id` | string | Task identifier |
| `predicted_threshold_min` | int | Minimum threshold rung meeting fidelity target |
| `predicted_forward_wall_s` | float | Runtime (seconds) for the 10,000-shot forward run |

Allowed threshold rungs:
```
1, 2, 4, 8, 16, 32, 64, 128, 256
```

Accepted JSON shapes:

**List**
```json
[
  { "id": "H001", "predicted_threshold_min": 16, "predicted_forward_wall_s": 15.8 }
]
```

**Wrapper**
```json
{
  "predictions": [
    { "id": "H001", "predicted_threshold_min": 16, "predicted_forward_wall_s": 15.8 }
  ]
}
```

---

## Local validation (recommended)

Use the validator to check format before submission:

```powershell
python scripts/validate_holdout_submission.py `
  --public data/holdout_public.json `
  --submission my_predictions.json `
  --write-normalized my_predictions.normalized.json
```

---

## Common pitfalls

- Uploading only `predict.py` when it imports `src.*`
- Forgetting to include trained model artifacts
- Hard-coding local file paths
- Nesting files inside an extra directory in the ZIP
- Assuming holdout task IDs reveal circuit identity

---

## Scoring summary

- Predicting a threshold **below** the true minimum scores **0**
- Predicting above the minimum is penalized
- Runtime accuracy is scored symmetrically
- Final ranking combines:
  - **Auto-grader Score**
  - **Presentation**
