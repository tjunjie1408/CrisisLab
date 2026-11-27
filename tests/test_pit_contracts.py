"""Unit tests for Approximate PIT, Label Maturity Embargo, and M&A Censoring invariants."""

from __future__ import annotations

import pandas as pd
import pandera.errors as pa_errors
import pytest

from crisislab.contracts.pit import (
    assert_no_temporal_leakage,
    attach_pit_timeline_and_labels,
    get_panel_contract_schema,
    split_with_label_maturity,
)


def test_reporting_lag_blocks_unreleased_q2_report_before_july_failure() -> None:
    """PDF Page 3 case: Bank fails on July 15; Q2 (June 30) Call Report is not yet public."""
    financials = pd.DataFrame(
        {
            "CERT": [101, 101],
            "REPDTE": ["2009-03-31", "2009-06-30"],
            "ASSET": [500_000.0, 480_000.0],
        }
    )
    failures = pd.DataFrame(
        {
            "CERT": [101],
            "FAILDATE": ["2009-07-15"],
            "RESTYPE": ["FAILURE"],
        }
    )

    labeled = attach_pit_timeline_and_labels(
        financials, failures, reporting_lag_days=60, horizon_days=365
    )
    q1, q2 = labeled.iloc[0], labeled.iloc[1]

    # Q1 (2009-03-31) becomes available 2009-05-30 -> before 2009-07-15 failure -> valid positive!
    assert q1["eligible_at_scoring_time"]
    assert q1["failed_in_horizon"] == 1

    # Q2 (2009-06-30) becomes available on 2009-08-29 -> AFTER 2009-07-15 failure -> ineligible!
    assert not q2["eligible_at_scoring_time"]
    assert q2["already_failed_by_available_date"]
    assert q2["failed_in_horizon"] == 0


@pytest.mark.parametrize(
    ("exit_date", "fail_date", "expected_failure"),
    [(20100530, "2010-05-31", 0), ("20100531", "2010-05-30", 1)],
)
def test_failed_charter_never_counts_as_non_failure_exit(
    exit_date: int | str, fail_date: str, expected_failure: int
) -> None:
    financials = pd.DataFrame({"CERT": [101], "REPDTE": [20090331]})
    failures = pd.DataFrame({"CERT": [101], "FAILDATE": [fail_date], "RESTYPE": ["FAILURE"]})
    institutions = pd.DataFrame({"CERT": [101], "INACTIVE": [1], "ENDEFYMD": [exit_date]})
    row = attach_pit_timeline_and_labels(financials, failures, institutions).iloc[0]
    assert row["eligible_at_scoring_time"]
    assert row["failed_in_horizon"] == expected_failure
    assert row["non_failure_exit_in_horizon"] == 0


def test_prior_insurance_lifecycle_does_not_disqualify_current_charter() -> None:
    financials = pd.DataFrame({"CERT": [101], "REPDTE": ["2009-03-31"]})
    failures = pd.DataFrame({"CERT": [101], "FAILDATE": ["1982-02-20"], "RESTYPE": ["FAILURE"]})
    institutions = pd.DataFrame(
        {
            "CERT": [101],
            "INACTIVE": [1],
            "INSDATE": ["1989-08-09"],
            "ENDEFYMD": ["2010-05-30"],
            "CHANGEC1": [223],
        }
    )
    row = attach_pit_timeline_and_labels(financials, failures, institutions).iloc[0]
    assert row["eligible_at_scoring_time"]
    assert row["failed_in_horizon"] == 0
    assert row["non_failure_exit_in_horizon"] == 1


def test_prior_insurance_lifecycle_handles_string_cert_in_institutions() -> None:
    financials = pd.DataFrame({"CERT": [101], "REPDTE": ["2009-03-31"]})
    failures = pd.DataFrame({"CERT": [101], "FAILDATE": ["1982-02-20"], "RESTYPE": ["FAILURE"]})
    institutions = pd.DataFrame(
        {
            "CERT": ["101"],
            "INACTIVE": [1],
            "INSDATE": ["1989-08-09"],
            "ENDEFYMD": ["2010-05-30"],
            "CHANGEC1": [223],
        }
    )
    row = attach_pit_timeline_and_labels(financials, failures, institutions).iloc[0]
    assert row["eligible_at_scoring_time"]
    assert row["failed_in_horizon"] == 0
    assert row["non_failure_exit_in_horizon"] == 1


def test_unlisted_bridge_resolution_is_excluded_from_non_failure_exits() -> None:
    financials = pd.DataFrame({"CERT": [101], "REPDTE": ["2009-03-31"]})
    institutions = pd.DataFrame(
        {"CERT": [101], "INACTIVE": [1], "ENDEFYMD": ["2010-01-13"], "CHANGEC1": [216]}
    )
    row = attach_pit_timeline_and_labels(financials, pd.DataFrame(), institutions).iloc[0]
    assert row["eligible_at_scoring_time"]
    assert row["failed_in_horizon"] == 0
    assert row["non_failure_exit_in_horizon"] == 0


def test_walk_forward_label_maturity_purges_2009q3_at_2010_jan_cutoff() -> None:
    """PDF Page 5-6: Training on 2010-01-01 cannot use 2009 Q3 (matures late 2010)."""
    financials = pd.DataFrame(
        {
            "CERT": [201, 201, 201],
            "REPDTE": ["2008-09-30", "2009-09-30", "2010-03-31"],
            "ASSET": [100_000.0, 95_000.0, 90_000.0],
        }
    )
    failures = pd.DataFrame(
        {
            "CERT": [201],
            "FAILDATE": ["2010-08-01"],
            "RESTYPE": ["FAILURE"],
        }
    )

    labeled = attach_pit_timeline_and_labels(
        financials, failures, reporting_lag_days=60, horizon_days=365
    )

    train_df, test_df = split_with_label_maturity(
        labeled,
        train_cutoff="2010-01-01",
        test_start="2010-01-01",
        test_end="2010-12-31",
    )

    # 2008-09-30 -> available 2008-11-29 -> matures 2009-11-29 <= 2010-01-01 -> kept in train (y=0)
    assert list(train_df["REPDTE"]) == [pd.Timestamp("2008-09-30")]
    assert list(train_df["failed_in_horizon"]) == [0]

    # 2009-09-30 -> available 2009-11-29 -> matures 2010-11-29 > 2010-01-01 -> PURGED from train!
    assert pd.Timestamp("2009-09-30") not in set(train_df["REPDTE"])

    # 2010-03-31 -> available 2010-05-30 -> in test set, fails 2010-08-01 -> y=1
    assert list(test_df["REPDTE"]) == [pd.Timestamp("2010-03-31")]
    assert list(test_df["failed_in_horizon"]) == [1]

    assert_no_temporal_leakage(train_df, test_df, train_cutoff="2010-01-01", reporting_lag_days=60)


def test_open_bank_assistance_excluded_from_primary_failure_label() -> None:
    """Open-Bank Assistance (RESTYPE='ASSISTANCE', e.g., BofA CERT 3510) is not a bank closure."""
    financials = pd.DataFrame(
        {"CERT": [3510], "REPDTE": ["2008-09-30"], "ASSET": [1_400_000_000.0]}
    )
    failures = pd.DataFrame(
        {"CERT": [3510], "FAILDATE": ["2009-01-16"], "RESTYPE": ["ASSISTANCE"]}
    )

    labeled = attach_pit_timeline_and_labels(
        financials, failures, include_open_bank_assistance=False
    )
    assert labeled.loc[0, "failed_in_horizon"] == 0
    assert labeled.loc[0, "assistance_in_horizon"] == 1


def test_primary_test_set_retains_future_ma_exits_without_lookahead_bias() -> None:
    """PDF Page 7 case: Banks acquired in the next 12m must NOT be dropped from primary test set."""
    financials = pd.DataFrame(
        {
            "CERT": [301, 302],
            "REPDTE": ["2011-03-31", "2011-03-31"],
            "ASSET": [200_000.0, 250_000.0],
        }
    )
    failures = pd.DataFrame(columns=["CERT", "FAILDATE", "RESTYPE"])
    institutions = pd.DataFrame(
        {
            "CERT": [301, 302],
            "INACTIVE": [1, 0],
            "ENDEFYMD": ["2011-10-15", None],
            "CHANGEC1": [223, None],
        }
    )

    labeled = attach_pit_timeline_and_labels(financials, failures, institutions)
    _, primary_test = split_with_label_maturity(
        labeled,
        train_cutoff="2011-01-01",
        test_start="2011-01-01",
        test_end="2011-12-31",
    )

    # Both banks were alive on 2011-05-30 (available_date); CERT 301 merged on 2011-10-15
    assert set(primary_test["CERT"]) == {301, 302}
    c301 = primary_test.loc[primary_test["CERT"] == 301].iloc[0]
    assert int(c301["non_failure_exit_in_horizon"]) == 1
    assert int(c301["failed_in_horizon"]) == 0


def test_duplicate_bank_quarter_rejected_by_contract() -> None:
    dup_df = pd.DataFrame(
        {
            "CERT": [101, 101],
            "REPDTE": pd.to_datetime(["2009-12-31", "2009-12-31"]),
            "ASSET": [100.0, 100.0],
        }
    )
    with pytest.raises(pa_errors.SchemaError):
        get_panel_contract_schema().validate(dup_df)
