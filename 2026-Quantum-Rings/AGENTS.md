# Repository Guidelines

## Project Structure & Module Organization
- `circuits/` — public OpenQASM 2.0 circuit files used for training and exploration.
- `data/` — JSON datasets: `hackathon_public.json` (labeled training data) and `holdout_public.json` (task list only).
- `docs/` — challenge specs, data schema, submission format, and circuit provenance.
- `scripts/` — utilities for validating and (organizer-only) scoring submissions.
- `src/` — starter space for your modeling code (currently empty).

## Build, Test, and Development Commands
- `python scripts/validate_holdout_submission.py --public data/holdout_public.json --submission my_predictions.json --write-normalized my_predictions.normalized.json`
  - Validates and normalizes your predictions file before upload.
- `python predict.py --tasks data/holdout_public.json --circuits circuits/ --id-map <ID_MAP_JSON> --out predictions.json`
  - Expected interface for your predictor (see `docs/SUBMISSION.md`).

## Coding Style & Naming Conventions
- Python-first repository; use 4-space indentation and PEP 8 naming (`snake_case` for functions/vars, `PascalCase` for classes).
- Prefer explicit, readable data processing steps over heavy metaprogramming.
- No formatter or linter is configured; if you add one, document the command in this file.

## Testing Guidelines
- There is no automated test suite yet.
- Treat `scripts/validate_holdout_submission.py` as a functional check for output format.
- If you add tests, keep them in `tests/` and name files `test_*.py`.

## Commit & Pull Request Guidelines
- The repo has no established commit message convention; use concise, imperative summaries (e.g., "Add feature extraction for QASM").
- PRs should include:
  - A brief description of the modeling approach or feature changes.
  - The command used to generate predictions.
  - Any new dependencies or artifacts (e.g., `requirements.txt`, `artifacts/`).

## Security & Configuration Tips
- Do not commit private holdout artifacts (hidden QASM or organizer truth files).
- Keep large model files in `artifacts/` and document how they are loaded.
- Avoid hard-coded absolute paths; use repo-relative paths like `data/` and `circuits/`.
