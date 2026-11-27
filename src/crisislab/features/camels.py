"""Numeric CAMELS features with exact backward quarter alignment."""

import numpy as np
import pandas as pd

from crisislab.contracts.pit import parse_fdic_dates

RATIOS = (
    "EQV", "RBC1AAJ", "RBCRWAJ", "NCLNLSR", "NTLNLSR", "LNATRESR",
    "EEFFR", "ROA", "ROE", "NIMY", "LNLSDEPR",
)
BALANCE_RATIOS = {
    "dep_to_asset": ("DEP", "ASSET"),
    "coredep_to_dep": ("COREDEP", "DEP"),
    "depunins_to_dep": ("DEPUNINS", "DEP"),
    "ore_to_asset": ("ORE", "ASSET"),
    "eq_to_asset": ("EQ", "ASSET"),
}
TREND_FIELDS = (
    "RBC1AAJ", "RBCRWAJ", "EQV", "NCLNLSR", "NTLNLSR", "LNATRESR",
    "ROA", "NIMY", "EEFFR", "ore_to_asset", "log_asset", "depunins_to_dep",
)


def build_feature_matrix(
    panel: pd.DataFrame, feature_set: str = "static_camels"
) -> pd.DataFrame:
    """Return model columns in panel row order with missing values preserved."""
    if feature_set not in {"static_camels", "camels_with_trends"}:
        raise ValueError(f"Unknown CAMELS feature set: {feature_set}")
    columns = [*RATIOS, "ASSET", "DEP", "COREDEP", "DEPUNINS", "ORE", "EQ"]
    numeric = panel[columns].apply(pd.to_numeric, errors="coerce").astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    static = numeric[list(RATIOS)].copy()
    static["log_asset"] = np.log1p(numeric["ASSET"].clip(lower=0))
    for name, (num, den) in BALANCE_RATIOS.items():
        static[name] = numeric[num] / numeric[den].where(numeric[den].gt(0))
    if feature_set == "static_camels":
        return static
    return _add_trends(panel, static)


def _add_trends(panel: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    dates = parse_fdic_dates(panel["REPDTE"])
    quarter = dates.dt.year * 4 + dates.dt.quarter - 1
    cert = panel["CERT"]
    keys = pd.MultiIndex.from_arrays([cert, quarter])
    history = static[list(TREND_FIELDS)].set_axis(keys)
    if not keys.is_unique:
        raise ValueError("CAMELS trends require unique CERT-quarter keys")

    def lag_keys(lag: int) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays([cert, quarter - lag])

    features = [static]
    for lag in (1, 4):
        previous = history.reindex(lag_keys(lag)).set_axis(static.index)
        features.append((static[list(TREND_FIELDS)] - previous).add_suffix(f"_d{lag}q"))

    extremes = {"RBC1AAJ": np.fmin, "NCLNLSR": np.fmax, "ROA": np.fmin}
    trailing = static[list(extremes)].copy()
    observed = np.ones(len(panel), dtype=int)
    for lag in (1, 2, 3):
        lookup = lag_keys(lag)
        observed += lookup.isin(keys)
        previous = history[list(extremes)].reindex(lookup)
        for field, reduce in extremes.items():
            trailing[field] = reduce(trailing[field].to_numpy(), previous[field].to_numpy())
    trailing.columns = ["RBC1AAJ_min4q", "NCLNLSR_max4q", "ROA_min4q"]
    trailing["quarters_observed"] = observed
    return pd.concat([*features, trailing], axis=1)

