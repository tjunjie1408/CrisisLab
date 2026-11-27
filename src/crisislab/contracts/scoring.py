"""Contracts for raw financial inputs and ranked quarterly watchlists."""

import numpy as np
import pandera.pandas as pa

from crisislab.features.camels import RATIOS

CORE_FIELDS = (*RATIOS, "ASSET", "DEP", "COREDEP", "DEPUNINS", "ORE", "EQ")
SORT_FIELDS = ["calibrated_prob", "raw_prob", "CERT"]
SORT_ASCENDING = [False, False, True]


def get_scoring_input_schema():
    return pa.DataFrameSchema(
        {
            "CERT": pa.Column(int, pa.Check.gt(0)),
            "REPDTE": pa.Column("datetime64[ns]"),
            **{field: pa.Column(nullable=True) for field in CORE_FIELDS},
            "ASSET": pa.Column(float, pa.Check.ge(0), nullable=True),
        },
        checks=pa.Check(lambda frame: not frame.duplicated(["CERT", "REPDTE"]).any()),
    )


def _ranked(frame):
    expected = frame.sort_values(SORT_FIELDS, ascending=SORT_ASCENDING)
    return (
        frame["REPDTE"].nunique() <= 1
        and np.array_equal(frame["risk_rank"], np.arange(1, len(frame) + 1))
        and frame["CERT"].equals(expected["CERT"])
    )


def get_watchlist_output_schema():
    return pa.DataFrameSchema(
        {
            "CERT": pa.Column(int, pa.Check.gt(0), unique=True),
            "NAME": pa.Column(str, nullable=True),
            "STALP": pa.Column(str, nullable=True),
            "REPDTE": pa.Column("datetime64[ns]"),
            "available_date": pa.Column("datetime64[ns]"),
            "raw_prob": pa.Column(float, pa.Check.in_range(0, 1)),
            "calibrated_prob": pa.Column(float, pa.Check.in_range(0, 1)),
            "risk_rank": pa.Column(int, pa.Check.ge(1)),
            "risk_percentile": pa.Column(float, [pa.Check.gt(0), pa.Check.le(100)]),
            "risk_tier": pa.Column(str, pa.Check.isin(["CRITICAL", "HIGH", "ELEVATED", "LOW"])),
            "in_top_2pct_budget": pa.Column(bool),
            "alert_fixed_p01": pa.Column(bool),
            "alert_fixed_p05": pa.Column(bool),
        },
        checks=pa.Check(_ranked),
    )

