"""Run the seven locked P1 ablations on an immutable FDIC snapshot."""

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd

from crisislab.config import PITConfig, ProjectConfig
from crisislab.contracts.pit import (
    CALIB_START,
    TEST_START,
    assert_three_way_no_leakage,
    attach_pit_timeline_and_labels,
    split_train_calib_test,
)
from crisislab.evaluation.metrics import evaluate_scores, slice_diagnostics
from crisislab.features.camels import build_feature_matrix
from crisislab.models.baselines import build_baseline, calibrated_probabilities
from crisislab.tracking.mlflow_tracker import log_panel_dataset_and_quality, start_data_centric_run

TARGET = "failed_in_horizon"
PRIMARY = "p1_lgbm_trends_4q60d"


@dataclass(frozen=True)
class Ablation:
    name: str
    model: str
    features: str
    horizon: int = 365
    lag: int = 60
    start_year: int = 2000
    censor_train: bool = False


ABLATIONS = (
    Ablation("p1_logreg_static_4q60d", "logreg", "static_camels"),
    Ablation("p1_lgbm_static_4q60d", "lgbm", "static_camels"),
    Ablation("p1_logreg_trends_4q60d", "logreg", "camels_with_trends"),
    Ablation(PRIMARY, "lgbm", "camels_with_trends"),
    Ablation("p1_lgbm_trends_4q60d_censor_train_ma", "lgbm", "camels_with_trends",
             censor_train=True),
    Ablation("p1_lgbm_trends_2q45d", "lgbm", "camels_with_trends", horizon=180, lag=45),
    Ablation("p1_lgbm_trends_4q60d_train2007", "lgbm", "camels_with_trends", start_year=2007),
)
SUMMARY = [
    "rows", "positive_rows", "pr_auc", "roc_auc", "brier_score", "ece_10bin",
    "recall_at_top_2pct", "event_recall_at_top_2pct", "median_lead_time_days_at_top_2pct",
]


def load_snapshot(raw: Path, cfg: ProjectConfig) -> tuple[pd.DataFrame, ...]:
    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    financials = f"financials_{cfg.panel.start_year}_{cfg.panel.end_year}"
    frames = []
    for key in (financials, "failures", "institutions"):
        entry = manifest[key]
        path = raw / entry["file"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Snapshot hash does not match manifest for {path}")
        frames.append(pd.read_parquet(path))
    return tuple(frames)


def run_ablation(
    variant: Ablation, cfg: ProjectConfig, raw: Path,
    panel: pd.DataFrame, features: pd.DataFrame,
) -> tuple[str, dict[str, pd.DataFrame]]:
    train, calib, test = split_train_calib_test(
        panel, censor_train_exits=variant.censor_train, train_start_year=variant.start_year,
    )
    assert_three_way_no_leakage(train, calib, test, reporting_lag_days=variant.lag)
    model = build_baseline(variant.model)
    with start_data_centric_run(
        cfg, experiment_name=cfg.tracking.p1_experiment_name, run_name=variant.name,
        stage="p1", manifest_path=raw / "manifest.json",
    ) as run:
        mlflow.log_params({
            "model": variant.model, "feature_set": variant.features,
            "train_start_year": variant.start_year, "censor_train_exits": variant.censor_train,
            "censor_test_exits_sensitivity_only": False,
            "calib_start": CALIB_START, "test_start": TEST_START,
            **{f"model.{key}": value for key, value in model.get_params().items()},
        })
        mlflow.log_dict({"columns": features.columns.tolist()}, "features.json")
        for name, part in zip(("train", "calib", "test"), (train, calib, test), strict=True):
            keys = ["CERT", "REPDTE", "available_date", "label_mature_date", TARGET,
                    "non_failure_exit_in_horizon"]
            log_panel_dataset_and_quality(
                part[keys], dataset_name=f"{variant.name}_{name}", context=name,
                source_label=str((raw / "manifest.json").resolve()),
            )
        model.fit(features.loc[train.index], train[TARGET])
        predictions = calibrated_probabilities(
            model.predict_proba(features.loc[calib.index])[:, 1], calib[TARGET].to_numpy(),
            model.predict_proba(features.loc[test.index])[:, 1], train_prior=train[TARGET].mean(),
        )
        comparison = pd.DataFrame.from_dict(
            {name: evaluate_scores(test, scores) for name, scores in predictions.items()},
            orient="index",
        )
        for name, values in comparison.iterrows():
            mlflow.log_metrics({f"{name}.{key}": float(value)
                                for key, value in values.items() if math.isfinite(value)})
        yearly, assets, sensitivity = slice_diagnostics(test, predictions["uncalibrated"])
        tables = {"calibration": comparison, "yearly": yearly,
                  "assets": assets, "exit_sensitivity": sensitivity}
        for name, table in tables.items():
            mlflow.log_text(table.to_csv(), f"{name}.csv")
        return run.info.run_id, tables


def write_report(output: Path, results: dict) -> None:
    summary = pd.DataFrame({
        name: tables["calibration"].loc["uncalibrated", SUMMARY].to_dict()
        | {"run_id": run_id} for name, (run_id, tables) in results.items()
    }).T
    lines = [
        "# P1 baseline and ablation report",
        "Approximate PIT uses public FDIC Call Reports, which include historical restatements.",
        "Test starts in 2013 by availability. December 2016 reports are scored in 2017.",
        "Top budgets use ceiling-rounded pooled rows and sort score ties by availability and CERT.",
        "Slices recompute alert budgets using fixed raw predictions.",
        "Prior adjustment starts from Platt probabilities and uses the historical training prior.",
        "The shorter-horizon ablation changes label and lag together.",
        "## Uncalibrated ablations", summary.to_markdown(floatfmt=".6g"),
    ]
    for name, (_, tables) in results.items():
        lines.append(f"## {name}")
        for table, frame in tables.items():
            lines.extend([f"### {table}", frame.to_markdown(floatfmt=".6g")])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    cfg = ProjectConfig.from_toml(args.config)
    if not cfg.tracking.enabled:
        parser.error("P1 requires MLflow tracking")
    snapshot = load_snapshot(args.raw_dir, cfg)
    results = {}
    feature_sets = {}
    current_formulation = None
    for variant in ABLATIONS:
        formulation = (variant.horizon, variant.lag)
        if formulation != current_formulation:
            cfg.pit = PITConfig(horizon_days=variant.horizon, reporting_lag_days=variant.lag)
            panel = attach_pit_timeline_and_labels(
                *snapshot[:2], institutions=snapshot[2],
                reporting_lag_days=variant.lag, horizon_days=variant.horizon,
            )
            current_formulation = formulation
        if variant.features not in feature_sets:
            feature_sets[variant.features] = build_feature_matrix(panel, variant.features)
        results[variant.name] = run_ablation(
            variant, cfg, args.raw_dir, panel, feature_sets[variant.features],
        )
        print(f"{variant.name} {results[variant.name][0]}", flush=True)
    output = args.output_dir / "p1_baseline_and_ablation_report.md"
    write_report(output, results)
    client = mlflow.MlflowClient()
    for run_id, _ in results.values():
        client.log_artifact(run_id, str(output))


if __name__ == "__main__":
    main()



