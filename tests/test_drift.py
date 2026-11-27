import numpy as np
import pandas as pd

from crisislab.monitoring.drift import compute_drift_report, compute_psi, fit_reference_bins


def test_psi_identical_shift_and_missing():
    train = pd.DataFrame({"feature": np.arange(100, dtype=float)})
    specs = fit_reference_bins(train, train.columns)
    spec = specs["feature"]
    assert compute_psi(spec, train["feature"]) == 0.0
    for current in [train + 100, train.mask(train < 80)]:
        drift, score = compute_drift_report(specs, current, current["feature"], spec)
        assert drift.loc[0, "status"] == "SIGNIFICANT_SHIFT"
        assert score["score_status"] == "SIGNIFICANT_SHIFT"
    assert drift.loc[0, "missing_delta_pct"] == 80


def test_zero_inflated_quantiles():
    train = pd.DataFrame({"ORE": [0.0] * 95 + [10.0] * 5})
    spec = fit_reference_bins(train, ["ORE"])["ORE"]
    assert spec["edges"] == [0.0]
    assert spec["proportions"] == [0.95, 0.05, 0.0]
    assert compute_psi(spec, train["ORE"]) == 0.0


def test_psi_value_and_missing_decomposition():
    train = pd.DataFrame({"feature": np.tile(np.arange(10, dtype=float), 10)})
    specs = fit_reference_bins(train, train.columns)
    spec = specs["feature"]

    def report(current):
        drift, _ = compute_drift_report(specs, current, current["feature"], spec)
        row = drift.iloc[0]
        assert np.isclose(row.value_psi + row.missing_psi, row.psi)
        return row

    shifted = report(train + 100)
    assert shifted.value_psi > 0
    assert shifted.missing_psi == 0

    missing = report(train.mask(train.index.to_series() >= 50))
    assert missing.missing_psi > 0
    # Uniform removal preserves conditional value frequencies but reduces finite bucket mass.
    assert np.isclose(missing.value_psi, 0.5 * np.log(2))
    assert missing.missing_psi > missing.value_psi
