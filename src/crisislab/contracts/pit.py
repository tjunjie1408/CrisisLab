"""Point-in-Time (PIT), label construction, and temporal split contracts."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

# FDIC structure changes for failure, assisted merger, bridge, and failure payoff.
FAILURE_EXIT_CODES = {211, 213, 215, 216, 217, 230}


def get_panel_contract_schema() -> pa.DataFrameSchema:
    """Data contract for standardized bank-quarter panels."""
    return pa.DataFrameSchema(
        columns={
            "CERT": pa.Column(int, nullable=False, checks=pa.Check.gt(0)),
            "REPDTE": pa.Column("datetime64[ns]", nullable=False),
            "ASSET": pa.Column(float, nullable=True, checks=pa.Check.ge(0)),
        },
        strict=False,
        unique=["CERT", "REPDTE"],
    )


def parse_fdic_dates(values: pd.Series, *, errors: str = "raise") -> pd.Series:
    """Parse compact YYYYMMDD values and the formatted dates FDIC returns."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    dates = values.astype("string").str.strip()
    compact = dates.str.fullmatch(r"[0-9]{8}", na=False)
    dates.loc[compact] = (
        dates.loc[compact].str.slice(0, 4)
        + "-"
        + dates.loc[compact].str.slice(4, 6)
        + "-"
        + dates.loc[compact].str.slice(6)
    )
    return pd.to_datetime(dates, format="mixed", errors=errors)


def attach_pit_timeline_and_labels(
    financials: pd.DataFrame,
    failures: pd.DataFrame,
    institutions: pd.DataFrame | None = None,
    *,
    reporting_lag_days: int = 60,
    horizon_days: int = 365,
    include_open_bank_assistance: bool = False,
) -> pd.DataFrame:
    """Construct Approximate PIT timeline, forward failure labels, and exit flags.

    Rules enforced:
    1. `available_date = REPDTE + reporting_lag_days`.
    2. If a bank already failed on or before `available_date` (`fail_date <= available_date`),
       that report was NOT available prior to failure (`already_failed_by_available_date = True`)
       and is excluded from eligible live scoring rows.
    3. `failed_in_horizon = 1` iff a true failure occurs in `(available_date, label_mature_date]`,
       where `label_mature_date = available_date + horizon_days`.
    4. Non-failure exits (e.g., unassisted M&A from `/institutions`) occurring in
       `(available_date, label_mature_date]` are flagged via `non_failure_exit_in_horizon`
       WITHOUT dropping rows or altering `failed_in_horizon` on evaluation sets.
    """
    df = financials.copy()
    df["CERT"] = df["CERT"].astype(int)
    df["REPDTE"] = parse_fdic_dates(df["REPDTE"])

    if df.duplicated(subset=["CERT", "REPDTE"]).any():
        dups = int(df.duplicated(subset=["CERT", "REPDTE"]).sum())
        raise ValueError(f"Duplicate (CERT, REPDTE) rows detected in financials: {dups}")

    lag = pd.Timedelta(days=reporting_lag_days)
    horizon = pd.Timedelta(days=horizon_days)

    df["available_date"] = df["REPDTE"] + lag
    df["label_mature_date"] = df["available_date"] + horizon

    # Prepare failure events per CERT (earliest relevant failure per CERT)
    failed_certs = pd.Index([], dtype=int)
    if failures.empty:
        fail_map = pd.DataFrame(columns=["CERT", "fail_date", "assistance_date"])
    else:
        f = failures.copy()
        f = f.dropna(subset=["CERT", "FAILDATE"]).copy()
        f["CERT"] = f["CERT"].astype(int)
        f["FAILDATE_DT"] = parse_fdic_dates(f["FAILDATE"])
        if institutions is not None and "INSDATE" in institutions and "CERT" in institutions:
            # A CERT can span insurance lifecycles. Ignore prior events when the
            # entire observed history for that CERT belongs to the current lifecycle.
            inst_cert = pd.to_numeric(institutions["CERT"], errors="coerce").astype("Int64")
            insured_since = parse_fdic_dates(institutions["INSDATE"], errors="coerce")
            first_report = inst_cert.map(df.groupby("CERT")["REPDTE"].min())
            current_lifecycle = insured_since.where(insured_since.le(first_report))
            started = f["CERT"].map(current_lifecycle.groupby(inst_cert).min())
            f = f.loc[~f["FAILDATE_DT"].lt(started)]
        default_res = pd.Series("FAILURE", index=f.index)
        restype = f.get("RESTYPE", default_res).fillna("FAILURE").astype(str).str.upper()
        failed_certs = pd.Index(f.loc[restype == "FAILURE", "CERT"])

        if include_open_bank_assistance:
            true_fail = f[restype.isin(["FAILURE", "ASSISTANCE"])]
        else:
            true_fail = f[restype == "FAILURE"]

        obam = f[restype == "ASSISTANCE"]

        fail_earliest = true_fail.groupby("CERT", as_index=False)["FAILDATE_DT"].min().rename(
            columns={"FAILDATE_DT": "fail_date"}
        )
        obam_earliest = obam.groupby("CERT", as_index=False)["FAILDATE_DT"].min().rename(
            columns={"FAILDATE_DT": "assistance_date"}
        )
        fail_map = pd.merge(fail_earliest, obam_earliest, on="CERT", how="outer")

    df = df.merge(fail_map, on="CERT", how="left")

    # Prepare non-failure structural exit dates from /institutions if provided
    if institutions is not None and not institutions.empty:
        inst = institutions.copy()
        inst = inst.dropna(subset=["CERT"]).copy()
        inst["CERT"] = inst["CERT"].astype(int)
        inactive = inst[inst.get("INACTIVE", 0) == 1].copy()
        if not inactive.empty and "ENDEFYMD" in inactive.columns:
            inactive["exit_date"] = parse_fdic_dates(inactive["ENDEFYMD"], errors="coerce")
            exit_cols = ["CERT", "exit_date"]
            if "CHANGEC1" in inactive.columns:
                exit_cols.append("CHANGEC1")
            inst_map = (
                inactive[exit_cols]
                .sort_values("exit_date")
                .drop_duplicates(subset=["CERT"], keep="last")
            )
            df = df.merge(inst_map, on="CERT", how="left")
        else:
            df["exit_date"] = pd.NaT
    else:
        df["exit_date"] = pd.NaT

    # Eligibility at scoring time `available_date`
    already_failed = df["fail_date"].notna() & (df["fail_date"] <= df["available_date"])
    already_exited = (
        df["exit_date"].notna()
        & (df["exit_date"] <= df["available_date"])
        & (~df["CERT"].isin(failed_certs))
    )
    df["already_failed_by_available_date"] = already_failed
    df["eligible_at_scoring_time"] = ~(already_failed | already_exited)

    # Primary binary target: FDIC failure strictly inside (available_date, label_mature_date]
    in_fail_window = (
        df["eligible_at_scoring_time"]
        & df["fail_date"].notna()
        & (df["fail_date"] > df["available_date"])
        & (df["fail_date"] <= df["label_mature_date"])
    )
    df["failed_in_horizon"] = in_fail_window.astype(int)

    # Open-bank assistance in horizon (tracked for diagnostic / sensitivity checks)
    in_obam_window = (
        df["eligible_at_scoring_time"]
        & df["assistance_date"].notna()
        & (df["assistance_date"] > df["available_date"])
        & (df["assistance_date"] <= df["label_mature_date"])
    )
    df["assistance_in_horizon"] = in_obam_window.astype(int)

    exit_codes = df.get("CHANGEC1", pd.Series(dtype="Int64", index=df.index))
    failure_resolution = pd.to_numeric(exit_codes, errors="coerce").isin(FAILURE_EXIT_CODES)
    # Competing risk flag: non-failure institution exit inside (available_date, label_mature_date]
    # The terminal closure of a failed charter is a failure regardless of date discrepancies.
    in_exit_window = (
        df["eligible_at_scoring_time"]
        & (~in_fail_window)
        & (~df["CERT"].isin(failed_certs))
        & (~failure_resolution)
        & df["exit_date"].notna()
        & (df["exit_date"] > df["available_date"])
        & (df["exit_date"] <= df["label_mature_date"])
    )
    df["non_failure_exit_in_horizon"] = in_exit_window.astype(int)

    return df


def split_with_label_maturity(
    labeled_panel: pd.DataFrame,
    *,
    train_cutoff: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
    censor_train_exits: bool = False,
    censor_test_exits_sensitivity_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create leakage-free train and test splits enforcing cohort label maturity.

    - Training split requires `label_mature_date <= train_cutoff` (purging any cohort whose
      forward prediction window crosses `train_cutoff`).
    - Test split selects cohorts whose `available_date` falls in `[test_start, test_end]`
      and keeps ALL eligible banks alive at `available_date` by default to prevent
      look-ahead M&A selection bias.
    """
    t_cut = pd.Timestamp(train_cutoff)
    t_start = pd.Timestamp(test_start)
    t_end = pd.Timestamp(test_end)

    if t_start < t_cut:
        raise ValueError(f"test_start ({t_start.date()}) must be >= train_cutoff ({t_cut.date()})")

    eligible = labeled_panel[labeled_panel["eligible_at_scoring_time"]].copy()

    # Train: both features available AND full forward horizon matured by train_cutoff
    train_mask = (eligible["available_date"] <= t_cut) & (eligible["label_mature_date"] <= t_cut)
    if censor_train_exits:
        train_mask = train_mask & (eligible["non_failure_exit_in_horizon"] == 0)

    # Test: features available during [test_start, test_end]
    test_mask = (eligible["available_date"] >= t_start) & (eligible["available_date"] <= t_end)
    if censor_test_exits_sensitivity_only:
        test_mask = test_mask & (eligible["non_failure_exit_in_horizon"] == 0)

    train_df = eligible.loc[train_mask].copy()
    test_df = eligible.loc[test_mask].copy()
    return train_df, test_df


def assert_no_temporal_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    train_cutoff: str | pd.Timestamp,
    reporting_lag_days: int = 60,
) -> None:
    """Explicit CI invariant check for temporal & PIT anti-leakage."""
    t_cut = pd.Timestamp(train_cutoff)
    min_lag = pd.Timedelta(days=reporting_lag_days)

    for name, part in [("train", train_df), ("test", test_df)]:
        if part.empty:
            continue
        actual_lag = part["available_date"] - part["REPDTE"]
        if (actual_lag < min_lag).any():
            msg = f"{name}: found available_date - REPDTE < {reporting_lag_days} days"
            raise AssertionError(msg)
        already_dead = part["fail_date"].notna() & (part["fail_date"] <= part["available_date"])
        if already_dead.any():
            raise AssertionError(f"{name}: found bank already failed on or before available_date")

    if not train_df.empty:
        if (train_df["label_mature_date"] > t_cut).any():
            raise AssertionError("train: found unmatured labels (label_mature_date > train_cutoff)")
        if not test_df.empty:
            max_mature = train_df["label_mature_date"].max()
            if max_mature > test_df["available_date"].min():
                raise AssertionError("train label_mature_date overlaps with test available_date")


CALIB_START = "2011-01-01"
TEST_START = "2013-01-01"


def split_train_calib_test(
    labeled_panel: pd.DataFrame,
    *,
    calib_start: str = CALIB_START,
    test_start: str = TEST_START,
    censor_train_exits: bool = False,
    train_start_year: int = 2000,
    censor_test_exits_sensitivity_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition eligible bank-quarters by availability, purging unmatured labels."""
    calib_cut, test_cut = pd.Timestamp(calib_start), pd.Timestamp(test_start)
    if not calib_cut < test_cut:
        raise ValueError("calib_start must precede test_start")
    eligible = labeled_panel.loc[labeled_panel["eligible_at_scoring_time"]]
    available = eligible["available_date"]
    mature = eligible["label_mature_date"]
    train = (
        available.lt(calib_cut) & mature.le(calib_cut)
        & eligible["REPDTE"].dt.year.ge(train_start_year)
    )
    calib = available.ge(calib_cut) & available.lt(test_cut) & mature.le(test_cut)
    test = available.ge(test_cut)
    if censor_train_exits:
        train &= eligible["non_failure_exit_in_horizon"].eq(0)
    if censor_test_exits_sensitivity_only:
        test &= eligible["non_failure_exit_in_horizon"].eq(0)
    return eligible.loc[train].copy(), eligible.loc[calib].copy(), eligible.loc[test].copy()


def assert_three_way_no_leakage(
    train: pd.DataFrame,
    calib: pd.DataFrame,
    test: pd.DataFrame,
    *,
    calib_start: str = CALIB_START,
    test_start: str = TEST_START,
    reporting_lag_days: int = 60,
) -> None:
    """Check eligibility, lag, maturity, availability bounds and disjoint row keys."""
    calib_cut, test_cut = pd.Timestamp(calib_start), pd.Timestamp(test_start)
    assert calib_cut < test_cut, "Invalid split boundaries"
    lag = pd.Timedelta(days=reporting_lag_days)
    for part in (train, calib, test):
        assert part["eligible_at_scoring_time"].eq(True).all(), "Ineligible bank-quarter"
        assert (part["available_date"] - part["REPDTE"]).eq(lag).all(), "Invalid PIT lag"
        assert (
            part["fail_date"].isna() | part["fail_date"].gt(part["available_date"])
        ).all(), "Bank already failed"
    assert train["available_date"].lt(calib_cut).all(), "Train availability overlap"
    assert train["label_mature_date"].le(calib_cut).all(), "Train embargo"
    assert calib["available_date"].ge(calib_cut).all(), "Calibration starts too early"
    assert calib["available_date"].lt(test_cut).all(), "Calibration availability overlap"
    assert calib["label_mature_date"].le(test_cut).all(), "Calibration embargo"
    assert test["available_date"].ge(test_cut).all(), "Test starts too early"
    keys = pd.concat([part[["CERT", "REPDTE"]] for part in (train, calib, test)])
    assert not keys.duplicated().any(), "Repeated bank-quarter key"

