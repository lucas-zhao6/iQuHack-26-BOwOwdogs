# Circuit Fingerprint Challenge (iQuHACK 2026)

Predict the simulation cost–accuracy tradeoffs of quantum circuits on an approximate quantum simulator, using Quantum Rings.

## Why this challenge matters

Quantum computing holds the promise to transform entire industries, but today, progress in quantum algorithms is constrained by our ability to execute and test large quantum circuits.

While quantum processing units (QPUs) continue to advance, most real-world development still happens in simulation. The limitation is fundamental: **perfect quantum simulation scales exponentially**, making it infeasible beyond modest circuit sizes.  To get past this, developers rely on **approximate simulation techniques** that generally trade accuracy for runtime and compute resources.

Modern simulators, like those developed by Quantum Rings, allow practitioners to push far beyond traditional limits by exposing configurable  parameters. However, this flexibility introduces a new challenge:

> *How do you know which simulation settings will deliver the fidelity you need—without wasting hours or days of compute?*

This challenge sits at the intersection of **quantum computing and machine learning**. Your task is to develop a model that predicts how a given quantum circuit will behave under different simulation parameters by extracting features from QASM structure, gate composition, entanglement patterns, and topology, and combining them with empirical performance data to **predict**:

- the *minimum approximation threshold* required to meet a target fidelity, and  
- the *expected runtime* to complete execution.

The outcome is quantitative, but the impact will be far more than a leaderboard score. Strong solutions may directly inform **real developer tooling** in an upcoming release, enabling faster iteration, smarter resource allocation, and more accessible quantum experimentation.

## Challenge Summary

In this challenge, you will build models that analyze **OpenQASM 2.0 quantum circuits** and predict how they will perform under different **approximate simulation configurations**. Given a circuit and execution context (CPU/GPU, precision), your goal is to estimate:

- the **minimum approximation threshold** required to meet a target fidelity, and  
- the **expected wall-clock runtime** for a fixed-shot forward simulation at that threshold.

Rather than running the simulator directly, you will infer performance from **circuit structure and prior empirical data**, using techniques from machine learning combined with well designed heuristics and feature engineering based on the input files.

Your predictions are evaluated against hidden holdout circuits using an accuracy-based scoring function, rewarding both **correct threshold selection** and **runtime estimation**. This mirrors a real-world developer workflow: choosing simulation settings that balance fidelity and compute time/cost before ever pressing “run.”

## Prizes & recognition

In addition to bragging rights and leaderboard placement, winning teams will receive:

- **$200 per person in free quantum compute credits** on **Open Quantum** (Maximum of $1,000 for the winning team)
- **Featured blog post** highlighting the winning team and their approach, with winners mentioned by name
- **LinkedIn post from Quantum Rings**, including photos of the winners, and a shout out by name
- **Guaranteed interview** for **Quantum Rings Quantum Internships (Summer 2026)**

Exceptional submissions may also receive honorable mentions, even if they do not place first overall.

## How the challenge works

This is an **offline prediction challenge**.

While you are free to experiment in any way you like, you are not required to run a quantum simulator during the hackathon. Instead, you will:

1. Study labeled performance data from a public set of quantum circuits
2. Extract features from circuit structure and metadata
3. Train a model or design heuristics to predict simulator behavior
4. Generate predictions for a set of hidden holdout circuits

Following team presentations, the hold out circuits will be provided, and the teams will run their code against the hidden circuits to produce predictions, which will then be evaluated against a private known truth file.

## Repository layout

The repository is organized to support the above workflow:

- `circuits/*.qasm` — curated **OpenQASM 2.0** circuits
- `data/hackathon_public.json` — **labeled** public dataset for training/validation
- `data/holdout_public.json` — holdout **task list** (IDs + CPU/GPU + precision). **Holdout QASM is not included.**
- `scripts/validate_holdout_submission.py` — checks that your predictions file is valid
- `scripts/score_holdout_submission.py` — scoring script (**requires a private truth file; organizers only**)

## One goal, one leaderboard

There is **one winner** based on a **single combined score**.

You will make predictions for **every task** in `data/holdout_public.json`.
Each task is one:

> task_id × (CPU/GPU) × (single/double precision)

(The circuit for each task is hidden; only organizers have the holdout QASM.)

For every task `id`, your submission must include **both**:

1) **`predicted_threshold_min`**  
   Your predicted *minimum* threshold (from the tested ladder) that will meet the fidelity target on the mirror benchmark.

2) **`predicted_forward_wall_s`**  
   Your predicted wall‑clock time (seconds) for the **10,000‑shot forward run** at that threshold.

The winner is the team with the **highest overall score**, computed by comparing your predictions to a private truth file (accuracy-based scoring). The score rewards:
- predicting the **minimum threshold rung** that meets mirror fidelity ≥ 0.99, and
- predicting the **10,000-shot forward runtime** at that rung.

See **`docs/SUBMISSION.md`** for the canonical format.

## Quick start

### 1) Explore the labeled dataset

Start with:

- `data/hackathon_public.json` (training data)
- `docs/DATA.md` (schema + interpretation)

### 2) Train a model (or build a heuristic baseline)

Feature ideas and modeling hints are in:

- `docs/CHALLENGE.md`

### 3) Predict the holdout tasks

Write a `predict.py` script that reads a tasks JSON and a circuits directory and writes a predictions JSON.

- Teams use the public training circuits for development.
- Organizers provide the hidden holdout circuits directory at scoring time.

See `docs/SUBMISSION.md` for the required interface and output format.

### 4) Validate your file locally

Use the validator to sanity-check your output format locally (optional debug artifact).

From the repo root:

```powershell
python scripts/validate_holdout_submission.py `
  --public data/holdout_public.json `
  --submission .\my_predictions.json `
  --write-normalized .\my_predictions.normalized.json
```

If validation succeeds, you may upload the normalized JSON as an optional debug artifact (not scored).

## Docs

Please find deeper documentation on the Challenge, the Data, the Submissions, and more, using the following links:

- `docs/CHALLENGE.md` — motivation, “what is threshold?”, and feature ideas
- `docs/DATA.md` — dataset format, fields, and interpretation
- `docs/SUBMISSION.md` — exact submission format + local validation instructions
- `docs/CIRCUITS.md` — circuit library notes + provenance guidance
- `docs/THIRD_PARTY.md` — third‑party attributions

## Support During the Hackathon

Rob Wamsley and Bob Wold will be available onsite as mentors during the daytime hacking sessions. During the nighttime hacking, Rob will be available via Discord until **10:00 PM**, then will return around **1:00 AM**.  

They will also be participating through both the official **Quantum Rings iQuHack Discord** and the **permanent Quantum Rings Discord server** throughout the hackathon.

For broader engagement with other Quantum Rings staff, or for a larger community of Quantum Rings users, during or as a follow-up after the hackathon, the Quantum Rings community will remain active.

You can join here: [https://discord.gg/uAzeWRAh86](https://discord.gg/uAzeWRAh86)

## License

This repository is MIT-licensed (see `LICENSE`).

Submissions (your code + write‑up) are separate; see `docs/SUBMISSION.md` for suggested licensing language.
