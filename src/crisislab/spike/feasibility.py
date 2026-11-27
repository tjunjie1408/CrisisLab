"""P0 Data Feasibility Spike: generates 4 empirical audit tables & logs Data-Centric MLflow run."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from crisislab.config import ProjectConfig
from crisislab.contracts.pit import (
    attach_pit_timeline_and_labels,
    get_panel_contract_schema,
    parse_fdic_dates,
)
from crisislab.data.fdic_client import FDICClient
from crisislab.tracking.mlflow_tracker import (
    log_panel_dataset_and_quality,
    start_data_centric_run,
)

# FDIC unassisted absorption, consolidation, merger, and partial purchase/assumption.
MA_EXIT_CODES = {221, 222, 223, 225}


def build_table1_panel_scale_and_missingness(
    panel: pd.DataFrame, camels_fields: list[str]
) -> pd.DataFrame:
    df = panel.copy()
    df["year"] = df["REPDTE"].dt.year
    camels = df.reindex(columns=camels_fields)

    records = []
    for year, grp in df.groupby("year"):
        miss = camels.loc[grp.index].isna().mean()
        records.append(
            {
                "year": int(year),
                "active_banks": int(grp["CERT"].nunique()),
                "bank_quarters": len(grp),
                "mean_camels_missing_pct": round(float(miss.mean() * 100), 2),
                "max_camels_missing_pct": round(float(miss.max() * 100), 2),
                "worst_missing_field": str(miss.idxmax()) if not miss.empty else "",
            }
        )
    return pd.DataFrame(records)


def build_table2_positive_events_vs_samples(labeled_panel: pd.DataFrame) -> pd.DataFrame:
    df = labeled_panel[labeled_panel["eligible_at_scoring_time"]].copy()
    df["feature_year"] = df["REPDTE"].dt.year

    records = []
    for year, grp in df.groupby("feature_year"):
        pos_rows = grp[grp["failed_in_horizon"] == 1]
        obam_rows = grp[grp["assistance_in_horizon"] == 1]
        records.append(
            {
                "feature_year": int(year),
                "eligible_bank_quarters": len(grp),
                "unique_banks": int(grp["CERT"].nunique()),
                "unique_failed_banks": int(pos_rows["CERT"].nunique()),
                "positive_bank_quarters": len(pos_rows),
                "positive_sample_rate_pct": round(len(pos_rows) / max(len(grp), 1) * 100, 3),
                "obam_banks": int(obam_rows["CERT"].nunique()),
            }
        )
    return pd.DataFrame(records)


def build_table3_non_failure_exits(labeled_panel: pd.DataFrame) -> pd.DataFrame:
    df = labeled_panel[labeled_panel["eligible_at_scoring_time"]].copy()
    df["feature_year"] = df["REPDTE"].dt.year

    records = []
    for year, grp in df.groupby("feature_year"):
        exits = grp[grp["non_failure_exit_in_horizon"] == 1]
        codes = exits.reindex(columns=["CHANGEC1"])["CHANGEC1"]
        ma_exits = exits.loc[pd.to_numeric(codes, errors="coerce").isin(MA_EXIT_CODES)]
        unique_banks = max(int(grp["CERT"].nunique()), 1)
        unique_exit_banks = int(exits["CERT"].nunique())
        ma_banks = int(ma_exits["CERT"].nunique())
        records.append(
            {
                "feature_year": int(year),
                "unique_banks": unique_banks,
                "non_failure_exit_banks": unique_exit_banks,
                "exit_bank_share_pct": round(unique_exit_banks / unique_banks * 100, 2),
                "exit_bank_quarters": len(exits),
                "ma_exit_banks": ma_banks,
                "ma_exit_bank_share_pct": round(ma_banks / unique_banks * 100, 2),
            }
        )
    return pd.DataFrame(records)


def build_exit_code_summary(labeled_panel: pd.DataFrame) -> pd.DataFrame:
    exits = labeled_panel.loc[labeled_panel["non_failure_exit_in_horizon"].eq(1)]
    codes = exits.reindex(columns=["CHANGEC1"])["CHANGEC1"].astype("Int64")
    codes = codes.astype("string").fillna("missing")
    return (
        exits.assign(CHANGEC1=codes)
        .groupby("CHANGEC1", as_index=False)
        .agg(exit_banks=("CERT", "nunique"), exit_bank_quarters=("CERT", "size"))
    )


def build_calendar_failure_counts(
    failures: pd.DataFrame, start_year: int, end_year: int
) -> pd.DataFrame:
    years = pd.Index(range(start_year, end_year + 1), name="FAILYR")
    return (
        pd.crosstab(pd.to_numeric(failures["FAILYR"]), failures["RESTYPE"])
        .reindex(index=years, columns=["FAILURE", "ASSISTANCE"], fill_value=0)
        .rename_axis(columns=None)
        .reset_index()
    )


def build_table4_single_bank_timeline_spotcheck(
    labeled_panel: pd.DataFrame, *, n_banks: int = 5, seed: int = 42
) -> pd.DataFrame:
    crisis_failed_certs = (
        labeled_panel.loc[
            labeled_panel["fail_date"].notna()
            & (labeled_panel["fail_date"].dt.year.between(2008, 2011)),
            "CERT",
        ]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )
    if len(crisis_failed_certs) == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        crisis_failed_certs, size=min(n_banks, len(crisis_failed_certs)), replace=False
    )

    sub = (
        labeled_panel[labeled_panel["CERT"].isin(chosen)]
        .sort_values(["CERT", "REPDTE"])
        .groupby("CERT")
        .tail(6)
        .copy()
    )
    cols = [
        c
        for c in [
            "CERT",
            "NAME",
            "REPDTE",
            "available_date",
            "label_mature_date",
            "fail_date",
            "eligible_at_scoring_time",
            "failed_in_horizon",
        ]
        if c in sub.columns
    ]
    return sub[cols].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CrisisLab P0 Data Feasibility Spike.")
    parser.add_argument("--config", default="configs/default.toml", help="Path to TOML config.")
    parser.add_argument("--start-year", type=int, default=None, help="Override start year.")
    parser.add_argument("--end-year", type=int, default=None, help="Override end year.")
    parser.add_argument("--lag-days", type=int, default=None, help="Override reporting lag days.")
    parser.add_argument("--horizon-days", type=int, default=None, help="Override horizon days.")
    parser.add_argument(
        "--include-obam",
        action="store_true",
        help="Include Open-Bank Assistance in primary failure label (sensitivity experiment).",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    parser.add_argument("--refresh", action="store_true", help="Re-download FDIC snapshots.")
    parser.add_argument(
        "--report-path",
        default="data/processed/p0_feasibility_report.md",
        help="Output Markdown summary path.",
    )
    args = parser.parse_args()

    cfg = ProjectConfig.from_toml(args.config)
    if args.start_year is not None:
        cfg.panel.start_year = args.start_year
    if args.end_year is not None:
        cfg.panel.end_year = args.end_year
    if args.lag_days is not None:
        cfg.pit.reporting_lag_days = args.lag_days
    if args.horizon_days is not None:
        cfg.pit.horizon_days = args.horizon_days
    if args.include_obam:
        cfg.pit.include_open_bank_assistance = True
    if args.no_mlflow:
        cfg.tracking.enabled = False

    start_year = cfg.panel.start_year
    end_year = cfg.panel.end_year

    client = FDICClient()
    print(f"[P0] Fetching FDIC failures, institutions, and financials ({start_year}-{end_year})...")
    failures = client.fetch_failures(refresh=args.refresh)
    institutions = client.fetch_institutions(refresh=args.refresh)
    financials = client.fetch_financials_panel(
        start_year=start_year,
        end_year=end_year,
        quarters=cfg.panel.quarters,
        camels_fields=cfg.panel.core_camels_fields,
        refresh=args.refresh,
    )

    financials["CERT"] = financials["CERT"].astype(int)
    financials["REPDTE"] = parse_fdic_dates(financials["REPDTE"]).astype("datetime64[ns]")
    for col in cfg.panel.core_camels_fields:
        financials[col] = pd.to_numeric(
            financials.get(col, pd.Series(np.nan, index=financials.index)), errors="coerce"
        ).astype(float)

    get_panel_contract_schema().validate(financials)

    labeled = attach_pit_timeline_and_labels(
        financials,
        failures,
        institutions,
        reporting_lag_days=cfg.pit.reporting_lag_days,
        horizon_days=cfg.pit.horizon_days,
        include_open_bank_assistance=cfg.pit.include_open_bank_assistance,
    )

    t1 = build_table1_panel_scale_and_missingness(labeled, cfg.panel.core_camels_fields)
    t1b = (
        labeled[cfg.panel.core_camels_fields]
        .isna()
        .groupby(labeled["REPDTE"].dt.year.rename("year"))
        .mean()
        .mul(100)
        .round(2)
        .reset_index()
    )
    calendar_counts = build_calendar_failure_counts(failures, start_year, end_year)
    exit_codes = build_exit_code_summary(labeled)
    t2 = build_table2_positive_events_vs_samples(labeled)
    t3 = build_table3_non_failure_exits(labeled)
    t4 = build_table4_single_bank_timeline_spotcheck(
        labeled, n_banks=5, seed=cfg.reproducibility.random_seed
    )

    out_path = Path(args.report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    include_obam = cfg.pit.include_open_bank_assistance
    sections = [
        f"# P0 Data Feasibility Spike Report ({start_year}–{end_year})\n",
        f"- Reporting Lag: `{cfg.pit.reporting_lag_days}` days (Approximate PIT)\n",
        f"- Prediction Horizon: `{cfg.pit.horizon_days}` days\n",
        f"- Include Open-Bank Assistance in Primary Label: `{include_obam}`\n",
        "\n## 1. Panel Scale & Core CAMELS Missingness by Year\n\n" + t1.to_markdown(index=False),
        "\n\n## 1B. Per-field CAMELS Missingness (%)\n\n" + t1b.to_markdown(index=False),
        "\n\n## Calendar-year FDIC Failure and Assistance Events\n\n"
        + calendar_counts.to_markdown(index=False),
        "\n\n## 2. Positive Failure Events vs. Bank-Quarter Samples\n\n"
        + t2.to_markdown(index=False),
        "\n\n## 3. Non-Failure Exit and M&A Shares\n\n"
        + t3.to_markdown(index=False),
        "\n\n## 3B. CHANGEC1 for Non-Failure Exits\n\n"
        + exit_codes.to_markdown(index=False),
        "\n\n## 4. Single-Bank Timeline Spot Check (5 Crisis Failures)\n\n"
        + t4.to_markdown(index=False)
        + "\n",
    ]
    report_md = "".join(sections)
    out_path.write_text(report_md, encoding="utf-8")

    run_name = (
        f"p0_{start_year}_{end_year}_lag{cfg.pit.reporting_lag_days}d"
        f"_h{cfg.pit.horizon_days}d_obam{int(include_obam)}"
    )
    with start_data_centric_run(
        cfg,
        experiment_name=cfg.tracking.p0_experiment_name,
        run_name=run_name,
        stage="p0_data_audit",
    ) as active_run:
        if active_run is not None:
            log_panel_dataset_and_quality(
                labeled,
                dataset_name=f"fdic_panel_{start_year}_{end_year}",
                context="p0_panel",
            )
            mlflow.log_metrics(
                {
                    "p0.mean_camels_missing_pct": float(t1["mean_camels_missing_pct"].mean())
                    if not t1.empty
                    else 0.0,
                    "p0.mean_exit_bank_share_pct": float(t3["exit_bank_share_pct"].mean())
                    if not t3.empty
                    else 0.0,
                    "p0.mean_ma_exit_bank_share_pct": float(t3["ma_exit_bank_share_pct"].mean())
                    if not t3.empty
                    else 0.0,
                }
            )
            mlflow.log_artifact(str(out_path), artifact_path="audit_reports")
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)
                for name, tbl in [
                    ("table1_panel_scale_missingness.csv", t1),
                    ("table1b_camels_missingness.csv", t1b),
                    ("calendar_failure_assistance_counts.csv", calendar_counts),
                    ("table3b_exit_codes.csv", exit_codes),
                    ("table2_positive_events_vs_samples.csv", t2),
                    ("table3_non_failure_exits.csv", t3),
                    ("table4_timeline_spotcheck.csv", t4),
                ]:
                    csv_path = tmp / name
                    tbl.to_csv(csv_path, index=False)
                    mlflow.log_artifact(str(csv_path), artifact_path="audit_tables")
            print(f"[P0] Logged Data-Centric MLflow run: {active_run.info.run_id} ({run_name})")

    print(f"[P0] Report written to {out_path}")


if __name__ == "__main__":
    main()
