"""Tests for Data-Centric MLflow lineage, dataset digest, and parameter tracking."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd

from crisislab.config import ProjectConfig
from crisislab.contracts.pit import attach_pit_timeline_and_labels
from crisislab.tracking.mlflow_tracker import (
    log_panel_dataset_and_quality,
    start_data_centric_run,
)


def test_data_centric_mlflow_logs_pit_params_snapshot_hashes_and_dataset(tmp_path: Path) -> None:
    db_path = tmp_path / "test_mlflow.db"
    art_dir = tmp_path / "mlartifacts"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "failures": {"file": "fdic_failures.parquet", "sha256": "abcdef1234567890fedcba"},
                "financials_2008_2010": {
                    "file": "fdic_financials_2008_2010.parquet",
                    "sha256": "9876543210abcdef123456",
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = ProjectConfig()
    cfg.pit.reporting_lag_days = 45
    cfg.pit.horizon_days = 180
    cfg.tracking.tracking_uri = f"sqlite:///{db_path.as_posix()}"
    cfg.tracking.artifact_root = str(art_dir)

    financials = pd.DataFrame(
        {
            "CERT": [101, 102],
            "REPDTE": ["2009-03-31", "2009-03-31"],
            "ASSET": [500_000.0, 750_000.0],
        }
    )
    failures = pd.DataFrame(
        {"CERT": [101], "FAILDATE": ["2009-08-01"], "RESTYPE": ["FAILURE"]}
    )
    labeled = attach_pit_timeline_and_labels(
        financials,
        failures,
        reporting_lag_days=cfg.pit.reporting_lag_days,
        horizon_days=cfg.pit.horizon_days,
    )

    with start_data_centric_run(
        cfg,
        experiment_name="test-data-centric-audit",
        run_name="audit_lag45_h180",
        stage="p0_data_audit",
        manifest_path=manifest_path,
    ) as active_run:
        assert active_run is not None
        run_id = active_run.info.run_id
        log_panel_dataset_and_quality(
            labeled,
            dataset_name="fdic_sample_panel",
            context="p0_panel",
        )

    fetched = mlflow.get_run(run_id)
    assert fetched.data.params["pit.reporting_lag_days"] == "45"
    assert fetched.data.params["pit.horizon_days"] == "180"
    assert fetched.data.tags["methodology"] == "data-centric-mle"
    assert fetched.data.tags["snapshot.failures.sha256"] == "abcdef1234567890fedcba"
    assert fetched.data.metrics["p0_panel.eligible_rows"] == 2.0
    assert fetched.data.metrics["p0_panel.positive_rows"] == 1.0
    assert len(fetched.inputs.dataset_inputs) == 1
