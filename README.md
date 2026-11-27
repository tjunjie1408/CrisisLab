# CrisisLab (CrisisML)

**Historical US Bank Failure Early Warning System** — An end-to-end, Data-Centric Machine Learning Engineering project built on public FDIC Call Report financials, failure records, and structural change history.

## Project Positioning

Instead of generic customer churn datasets, CrisisLab frames quarterly bank distress prediction around the 2008–2010 Global Financial Crisis and its 2011–2014 aftermath, focusing on production-grade **ML Engineering rigor**:

1. **Approximate Point-in-Time (PIT) Discipline**: Enforces reporting lag (`repdte + 60 days`) before features become eligible for scoring, explicitly documenting Call Report restatements as a known limitation rather than claiming strict real-time archival reconstruction.
2. **Label Maturity & Walk-Forward Embargo**: Purges unmatured forward windows (`available_date + 365 days > train_cutoff`) from training and calibration splits to eliminate subtle horizon look-ahead leakage.
3. **Competing Risks & Zero Look-Ahead Selection Bias**: Distinguishes true FDIC closures (`RESTYPE == "FAILURE"`) from Open-Bank Assistance (`RESTYPE == "ASSISTANCE"`) and unassisted M&A exits (`CHANGEC1`), keeping all banks active at scoring time in primary test evaluation.
4. **Deterministic Reproducibility & Data-Centric MLflow**: Rather than only logging model hyperparameters, MLflow is wired from **day one** (`sqlite:///mlflow.db`) to track **data & label experiments** as first-class variables (`reporting_lag_days`, `horizon_days`, M&A censoring, OBAM label boundary, raw snapshot SHA-256 digests, and `mlflow.data` dataset inputs).

## Delivery Roadmap (Data-Centric V3 Consensus)

| Component | Scope | Acceptance Criteria |
| :--- | :--- | :--- |
| **Data Audit** | **FDIC Snapshot & Feasibility Audit** | Pull FDIC snapshots, log `mlflow.data` lineage & SHA-256 manifest, output 4 empirical audit tables |
| **Baselines & Ablations** | **Data-Centric Feature & Model Ablations** | Fix LogReg + LightGBM (`deterministic=True`), run MLflow ablations over horizon/lag/censoring & calibration |
| **Batch Scoring & CI** | **Quarterly Scoring Pipeline & Contracts** | Single-command quarterly scoring & report generation; Pandera schema & pytest PIT leakage guards |
| **Registry & Docker** | **Model Registry & Container Packaging** | Promote calibrated pipeline in MLflow Registry & run reproducible containerized batch scoring |
| **Sensitivity API (Optional)** | **Drift Diagnostics & Scenario Sensitivity** | Post-crisis false-positive drift analysis; optional Scenario Sensitivity API (not causal stress testing) |

## Verified historical findings

The LightGBM baseline with temporal CAMELS features achieved a PR-AUC of **0.2881**,
event recall of **96.7%**, and a median lead time of **297 days** on the out-of-time
cohort. In the 2013Q4 watchlist, **6 of the top 15 banks (40%)** failed within the
365-day horizon. These are retrospective results under Approximate PIT.

## Quickstart

```powershell
# Install dependencies in pinned Python 3.12 virtualenv
uv sync

# Run anti-leakage, PIT contract & MLflow lineage unit tests
uv run pytest

# Run Data Feasibility Audit (downloads FDIC data & logs audit run to MLflow)
uv run crisislab-audit --start-year 2000 --end-year 2016

# Compare alternative data/label assumptions in MLflow (e.g., 2Q vs 4Q horizon, 45d vs 60d lag)
uv run crisislab-audit --start-year 2000 --end-year 2016 --horizon-days 180 --lag-days 45

# Run the controlled baseline ablations against the verified snapshot
uv run crisislab-experiments

# Fit and promote the production model bundle to the champion alias
uv run crisislab-register

# Replay a quarter with the registered bundle and its Train PSI references
uv run crisislab-score --quarter 2013Q4 --model-uri models:/crisislab-bank-failure-ews@champion --output-dir data/processed/batch_scoring_registry

# Launch local MLflow UI to inspect Data-Centric runs, dataset digests, and audit tables
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

See [docs/proposal-analysis-and-roadmap.md](docs/proposal-analysis-and-roadmap.md) for the full design rationale and empirical API notes.

## Docker

```powershell
docker build -t crisislab:local .
docker run --rm crisislab:local crisislab-score --help
docker run --rm crisislab:local crisislab-register --help
docker run --rm --mount "type=bind,source=${PWD}/data,target=/app/data" crisislab:local crisislab-score --quarter 2013Q4
```

The bind mount supplies the cached FDIC snapshot and preserves scoring output. To use
a registry from a container, configure a tracking URI and artifact store accessible from
that container. Local Windows artifact paths cannot be resolved inside a Linux container.

Each drift report breaks total PSI into finite-bin and missing-bin contributions.
If missingness rises, finite-bin contributions can rise too because those bins hold a
smaller share of all rows even when the distribution among finite values stays fixed.
