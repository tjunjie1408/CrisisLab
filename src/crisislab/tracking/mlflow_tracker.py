"""Data-Centric MLflow experiment tracking and dataset lineage utilities."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import mlflow.data
import pandas as pd

from crisislab.config import ProjectConfig


def get_git_commit_short() -> str:
    """Return short git commit hash if available, else 'uncommitted'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        sha = res.stdout.strip()
        return sha if res.returncode == 0 and sha else "uncommitted"
    except Exception:
        return "uncommitted"


def configure_mlflow(tracking_uri: str, experiment_name: str, artifact_root: str) -> str:
    """Configure MLflow tracking URI and ensure experiment exists with artifact root."""
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        art_uri = Path(artifact_root).resolve().as_uri()
        return mlflow.create_experiment(name=experiment_name, artifact_location=art_uri)
    return exp.experiment_id


@contextmanager
def start_data_centric_run(
    cfg: ProjectConfig,
    *,
    experiment_name: str,
    run_name: str,
    stage: str,
    manifest_path: str | Path = "data/raw/manifest.json",
    extra_tags: dict[str, str] | None = None,
) -> Iterator[mlflow.ActiveRun | None]:
    """Start an MLflow run pre-populated with PIT/label formulation params & snapshot hashes.

    In a Data-Centric workflow, data & label assumptions (reporting lag, horizon, M&A/OBAM
    treatment, snapshot SHA-256 digests) are first-class experimental variables, not static
    afterthoughts.
    """
    if not cfg.tracking.enabled:
        yield None
        return

    configure_mlflow(
        tracking_uri=cfg.tracking.tracking_uri,
        experiment_name=experiment_name,
        artifact_root=cfg.tracking.artifact_root,
    )
    mlflow.set_experiment(experiment_name)

    tags: dict[str, str] = {
        "methodology": "data-centric-mle",
        "stage": stage,
        "pit_mode": "approximate_pit",
        "git_commit": get_git_commit_short(),
    }
    if extra_tags:
        tags.update(extra_tags)

    m_path = Path(manifest_path)
    manifest_data: dict[str, Any] = {}
    if m_path.exists():
        manifest_data = json.loads(m_path.read_text(encoding="utf-8"))
        for ds_key, meta in manifest_data.items():
            if isinstance(meta, dict) and "sha256" in meta:
                tags[f"snapshot.{ds_key}.sha256"] = str(meta["sha256"])

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        mlflow.log_params(
            {
                "pit.reporting_lag_days": cfg.pit.reporting_lag_days,
                "pit.horizon_days": cfg.pit.horizon_days,
                "pit.include_open_bank_assistance": cfg.pit.include_open_bank_assistance,
                "panel.start_year": cfg.panel.start_year,
                "panel.end_year": cfg.panel.end_year,
                "panel.n_camels_fields": len(cfg.panel.core_camels_fields),
                "repro.random_seed": cfg.reproducibility.random_seed,
                "repro.lgbm_deterministic": cfg.reproducibility.lightgbm_deterministic,
                "repro.lgbm_force_col_wise": cfg.reproducibility.lightgbm_force_col_wise,
            }
        )
        if m_path.exists():
            mlflow.log_artifact(str(m_path), artifact_path="data_lineage")
        yield run


def log_panel_dataset_and_quality(
    panel: pd.DataFrame,
    *,
    dataset_name: str,
    context: str,
    source_label: str = "fdic_bankfind_api",
    target_col: str | None = "failed_in_horizon",
) -> dict[str, float]:
    """Log an `mlflow.data` PandasDataset input plus summary data-health metrics."""
    kwargs: dict[str, Any] = {"source": source_label, "name": dataset_name}
    if target_col and target_col in panel.columns:
        kwargs["targets"] = target_col

    dataset = mlflow.data.from_pandas(panel, **kwargs)
    mlflow.log_input(dataset, context=context)

    eligible = (
        panel[panel["eligible_at_scoring_time"]]
        if "eligible_at_scoring_time" in panel.columns
        else panel
    )
    metrics: dict[str, float] = {
        f"{context}.total_rows": float(len(panel)),
        f"{context}.eligible_rows": float(len(eligible)),
        f"{context}.unique_banks": float(eligible["CERT"].nunique()) if "CERT" in eligible else 0.0,
    }
    if target_col and target_col in eligible.columns and len(eligible) > 0:
        pos = eligible[eligible[target_col] == 1]
        metrics[f"{context}.positive_rows"] = float(len(pos))
        metrics[f"{context}.positive_rate_pct"] = float(len(pos) / len(eligible) * 100.0)
        if "CERT" in pos.columns:
            metrics[f"{context}.unique_positive_banks"] = float(pos["CERT"].nunique())
    if "non_failure_exit_in_horizon" in eligible.columns and len(eligible) > 0:
        exits = eligible[eligible["non_failure_exit_in_horizon"] == 1]
        metrics[f"{context}.non_failure_exit_rows"] = float(len(exits))

    mlflow.log_metrics(metrics)
    return metrics
