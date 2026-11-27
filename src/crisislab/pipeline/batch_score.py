"""Replay quarterly failure watchlists under an Approximate PIT cutoff."""

import argparse
import json
import math
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from crisislab.config import ProjectConfig
from crisislab.contracts.pit import (
    TEST_START,
    assert_three_way_no_leakage,
    attach_pit_timeline_and_labels,
    parse_fdic_dates,
    split_train_calib_test,
)
from crisislab.contracts.scoring import (
    SORT_ASCENDING,
    SORT_FIELDS,
    get_scoring_input_schema,
    get_watchlist_output_schema,
)
from crisislab.features.camels import build_feature_matrix
from crisislab.models.bundle import CalibratedFailureBundle
from crisislab.models.registry import _load_bundle_artifact
from crisislab.monitoring.drift import compute_drift_report, fit_reference_bins
from crisislab.pipeline.experiments import TARGET, load_snapshot
from crisislab.tracking.mlflow_tracker import log_panel_dataset_and_quality, start_data_centric_run

LIMITATION = "Approximate PIT. Public FDIC Call Reports contain historical restatements."
SPOTLIGHT = ["depunins_to_dep", "depunins_to_dep_d4q", "ORE", "NCLNLSR", "NIMY",
             "ore_to_asset_d4q", "NCLNLSR_d4q", "NIMY_d4q"]


def _as_of_events(failures, institutions, as_of):
    if not failures.empty:
        failures = failures.loc[parse_fdic_dates(failures["FAILDATE"]).le(as_of)]
    if not institutions.empty:
        institutions = institutions.copy()
        exits = parse_fdic_dates(institutions["ENDEFYMD"], errors="coerce")
        institutions.loc[~exits.le(as_of), "INACTIVE"] = 0
        if "INSDATE" in institutions:
            starts = parse_fdic_dates(institutions["INSDATE"], errors="coerce")
            institutions.loc[starts.gt(as_of), "INSDATE"] = None
    return failures, institutions


def _monitor_features(financials, features):
    return features.assign(ORE=pd.to_numeric(financials["ORE"], errors="coerce"))


def score_quarter(snapshot, target, bundle, cfg):
    financials, failures, institutions = snapshot
    dates = parse_fdic_dates(financials["REPDTE"])
    history = financials.loc[dates.le(target)].copy().reset_index(drop=True)
    history["REPDTE"] = dates.loc[dates.le(target)].to_numpy()
    history["ASSET"] = pd.to_numeric(history["ASSET"]).astype(float)
    get_scoring_input_schema().validate(history)
    as_of = target + pd.Timedelta(days=cfg.pit.reporting_lag_days)
    failures, institutions = _as_of_events(failures, institutions, as_of)
    timeline = attach_pit_timeline_and_labels(
        history, failures, institutions, reporting_lag_days=cfg.pit.reporting_lag_days,
        horizon_days=cfg.pit.horizon_days,
    )
    cross = timeline.loc[
        timeline["REPDTE"].eq(target) & timeline["eligible_at_scoring_time"]
    ]
    if cross.empty:
        raise ValueError(f"No eligible financial reports for {target.date()}")
    features = build_feature_matrix(timeline, cfg.scoring.feature_set).loc[cross.index]
    raw, calibrated = bundle.predict(features)
    watch = cross.reindex(columns=["CERT", "NAME", "STALP", "REPDTE", "available_date"]).copy()
    for field in ("NAME", "STALP"):
        watch[field] = watch[field].astype("string")
    watch = watch.assign(raw_prob=raw, calibrated_prob=calibrated)
    watch = watch.sort_values(SORT_FIELDS, ascending=SORT_ASCENDING).reset_index(drop=True)
    watch["risk_rank"] = np.arange(1, len(watch) + 1)
    watch["risk_percentile"] = 100.0 * (len(watch) + 1 - watch["risk_rank"]) / len(watch)
    p = watch["calibrated_prob"]
    settings = cfg.scoring
    watch["risk_tier"] = np.select(
        [p.ge(settings.tier_critical_threshold), p.ge(settings.tier_high_threshold),
         p.ge(settings.tier_elevated_threshold)],
        ["CRITICAL", "HIGH", "ELEVATED"], default="LOW",
    )
    watch["in_top_2pct_budget"] = watch["risk_rank"].le(
        math.ceil(len(watch) * settings.top_budget_pct / 100),
    )
    watch["alert_fixed_p01"] = p.ge(0.01)
    watch["alert_fixed_p05"] = p.ge(0.05)
    get_watchlist_output_schema().validate(watch)
    drift, score_drift = compute_drift_report(
        bundle.reference_feature_bins, _monitor_features(cross, features), calibrated,
        bundle.reference_score_bins, moderate=settings.psi_moderate_threshold,
        significant=settings.psi_significant_threshold,
    )
    return watch, drift, score_drift


def _split_snapshot(snapshot, cfg):
    panel = attach_pit_timeline_and_labels(
        *snapshot, reporting_lag_days=cfg.pit.reporting_lag_days,
        horizon_days=cfg.pit.horizon_days,
    )
    train, calib, test = split_train_calib_test(panel)
    assert_three_way_no_leakage(
        train, calib, test, reporting_lag_days=cfg.pit.reporting_lag_days,
    )
    return panel, train, calib, test


def _fit_bundle(snapshot, cfg, hashes):
    panel, train, calib, test = _split_snapshot(snapshot, cfg)
    # Only reports through the last calibration quarter enter bundle feature construction.
    history = panel.loc[panel["REPDTE"].le(calib["REPDTE"].max())]
    features = build_feature_matrix(history, cfg.scoring.feature_set)
    train_features = features.loc[train.index]
    bundle = CalibratedFailureBundle.fit(
        train_features, train[TARGET], features.loc[calib.index], calib[TARGET], hashes,
        n_bins=cfg.scoring.psi_bins,
    )
    bundle.reference_feature_bins = fit_reference_bins(
        _monitor_features(train, train_features),
        [*bundle.feature_names, "ORE"], cfg.scoring.psi_bins,
    )
    return bundle, test, train_features


def _summarize(quarter, watch, drift, score_drift, outcomes):
    labels = watch[["CERT", "REPDTE"]].merge(
        outcomes[["CERT", "REPDTE", TARGET]], on=["CERT", "REPDTE"], validate="one_to_one",
        how="left",
    )[TARGET]
    if labels.isna().any():
        raise ValueError(f"Unmatched retrospective outcome for {quarter}")
    negative = labels.eq(0)
    positive = labels.eq(1)
    top = drift.sort_values(["psi", "feature"], ascending=[False, True]).iloc[0]
    return {
        "quarter": quarter, "available_date": watch["available_date"].iloc[0].date(),
        "banks_scored": len(watch), "mean_calibrated_prob": watch["calibrated_prob"].mean(),
        "alert_rate_p01_pct": 100 * watch["alert_fixed_p01"].mean(),
        "alert_rate_p05_pct": 100 * watch["alert_fixed_p05"].mean(),
        "top_2pct_count": int(watch["in_top_2pct_budget"].sum()), **score_drift,
        "n_features_significant_drift": int(drift["status"].eq("SIGNIFICANT_SHIFT").sum()),
        "top_drifting_feature": top["feature"], "top_drifting_psi": top["psi"],
        "realized_failures_4q": int(positive.sum()),
        "fixed_p01_fpr_pct": 100 * watch.loc[negative, "alert_fixed_p01"].mean(),
        "top_2pct_recall": watch.loc[positive, "in_top_2pct_budget"].mean(),
    }


def _write_report(path, sections):
    path.write_text("\n\n".join([LIMITATION, *sections]) + "\n", encoding="utf-8")


def _table(frame):
    return frame.to_markdown(index=False, floatfmt=".5g")


def _insurance_comparison(financials):
    dates = parse_fdic_dates(financials["REPDTE"])
    window = financials.loc[dates.isin(pd.to_datetime(["2008-09-30", "2008-12-31"]))]
    features = build_feature_matrix(window)
    before = features.loc[dates.loc[window.index].dt.quarter.eq(3)]
    after = features.loc[dates.loc[window.index].dt.quarter.eq(4)]
    field = "depunins_to_dep"
    if before.empty or after.empty:
        return "The snapshot lacks one or both quarters for the October 2008 comparison."
    spec = fit_reference_bins(before, [field])
    drift, _ = compute_drift_report(spec, after, after[field], spec[field])
    return _table(drift)


def batch_score(
    snapshot, cfg, output, manifest_path, *, quarter=None, model_uri: str | None = None,
):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = {key: entry["sha256"] for key, entry in manifest.items() if "sha256" in entry}
    output.mkdir(parents=True, exist_ok=True)
    bundle_dir = output / "bundle"
    if model_uri is None:
        bundle, test, train_features = _fit_bundle(snapshot, cfg, hashes)
        bundle.save(bundle_dir)
        reloaded = CalibratedFailureBundle.load(bundle_dir)
        probe = train_features.head(100)
        for expected, observed in zip(bundle.predict(probe), reloaded.predict(probe), strict=True):
            np.testing.assert_array_equal(expected.view(np.uint64), observed.view(np.uint64))
    else:
        mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
        reloaded, bundle_dir = _load_bundle_artifact(model_uri)
        _, _, _, test = _split_snapshot(snapshot, cfg)
    metadata_path = bundle_dir / "bundle_metadata.json"
    financials = snapshot[0]
    dates = parse_fdic_dates(financials["REPDTE"])
    available = dates + pd.Timedelta(days=cfg.pit.reporting_lag_days)
    targets = (
        [pd.Period(quarter, freq="Q").end_time.normalize()] if quarter else
        sorted(dates.loc[available.ge(pd.Timestamp(TEST_START))].unique())
    )
    if not targets or min(targets) + pd.Timedelta(
        days=cfg.pit.reporting_lag_days
    ) < pd.Timestamp(TEST_START):
        raise ValueError("Scoring requires a quarter in the out-of-time period")
    summaries, spotlights, artifacts, scored_keys = [], [], [], []
    example = None
    for target in targets:
        target = pd.Timestamp(target)
        key = str(target.to_period("Q"))
        watch, drift, score_drift = score_quarter(snapshot, target, reloaded, cfg)
        row = _summarize(key, watch, drift, score_drift, test)
        summaries.append(row)
        spotlights.append(drift.loc[drift["feature"].isin(SPOTLIGHT)].assign(quarter=key))
        scored_keys.append(watch[["CERT", "REPDTE", "available_date"]])
        watch_path = output / f"watchlist_{key}.csv"
        drift_path = output / f"drift_{key}.csv"
        report = output / f"report_{key}.md"
        watch.to_csv(watch_path, index=False)
        drift.to_csv(drift_path, index=False)
        _write_report(report, [
            f"# {key} scoring", "Feature and calibrated score PSI use the Train cohort.",
            _table(pd.DataFrame([row]).drop(
                columns=["realized_failures_4q", "fixed_p01_fpr_pct", "top_2pct_recall"],
            )), "## Top 15", _table(watch.head(15)),
        ])
        artifacts.extend([watch_path, drift_path, report])
        if example is None or key == "2013Q4":
            example = (key, watch.head(15))
    summary = pd.DataFrame(summaries)
    if quarter is None:
        summary_path = output / "quarterly_summary.csv"
        summary.to_csv(summary_path, index=False)
        report = output.parent / "p2_batch_scoring_and_drift_report.md"
        _write_report(report, [
            "# P2 quarterly batch scoring and drift",
            "Feature and calibrated score PSI use Train as the reference cohort.",
            "Realized failures, false-positive rates and budget recall are retrospective "
            "bank-quarter checks. Undefined rates are shown as NaN.",
            _table(summary),
            "## CAMELS and trend spotlight",
            _table(pd.concat(spotlights, ignore_index=True)),
            "## October 2008 uninsured deposit reporting comparison",
            "The $100,000 to $250,000 insurance cap change can affect the measured ratio. "
            "This descriptive comparison uses 2008Q3 as its reference and 2008Q4 as current. "
            "Quarterly cross-sections also reflect changes in bank composition and funding.",
            _insurance_comparison(financials),
            f"## {example[0]} top 15 for triage", _table(example[1]),
        ])
        artifacts.extend([summary_path, report])
    with start_data_centric_run(
        cfg, experiment_name=cfg.tracking.p2_experiment_name, stage="p2",
        run_name=quarter or "all-test-quarters", manifest_path=manifest_path,
    ) as run:
        if run is None:
            return summary, None
        mlflow.log_params(cfg.scoring.model_dump())
        if model_uri is not None:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            mlflow.log_params({"model_uri": model_uri, "bundle.sha256": metadata["sha256"]})
        log_panel_dataset_and_quality(
            pd.concat(scored_keys, ignore_index=True), dataset_name="scored_bank_quarters",
            context="scoring", source_label=str(manifest_path.resolve()), target_col=None,
        )
        mlflow.log_metrics({
            "quarters_scored": len(summary),
            "mean_score_psi": summary["score_psi"].mean(),
            "max_score_psi": summary["score_psi"].max(),
            "mean_alert_rate_p01_pct": summary["alert_rate_p01_pct"].mean(),
            "mean_alert_rate_p05_pct": summary["alert_rate_p05_pct"].mean(),
            "mean_significant_features": summary["n_features_significant_drift"].mean(),
        })
        mlflow.log_artifact(str(metadata_path), artifact_path="bundle")
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
        return summary, run.info.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--quarter")
    modes.add_argument("--all-test-quarters", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/batch_scoring"))
    parser.add_argument("--model-uri")
    args = parser.parse_args()
    if args.quarter:
        import re

        if not re.fullmatch(r"[0-9]{4}Q[1-4]", args.quarter):
            parser.error("--quarter must use YYYYQ1 through YYYYQ4 format")
    cfg = ProjectConfig.from_toml(args.config)
    if cfg.pit.include_open_bank_assistance:
        parser.error("P2 requires the failure-only target")
    summary, run_id = batch_score(
        load_snapshot(args.raw_dir, cfg), cfg, args.output_dir, args.raw_dir / "manifest.json",
        quarter=args.quarter, model_uri=args.model_uri,
    )
    print(f"Scored {len(summary)} quarter(s). MLflow run {run_id}.")


if __name__ == "__main__":
    main()
