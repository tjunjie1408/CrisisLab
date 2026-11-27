"""Population stability with an explicit missing bucket."""

import numpy as np
import pandas as pd


def _distribution(series, edges):
    values = series.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(values)
    clean = values[valid]
    buckets = np.searchsorted(edges, clean, side="left")
    counts = np.r_[np.bincount(buckets, minlength=len(edges) + 1), (~valid).sum()]
    if not len(values):
        raise ValueError("PSI requires a nonempty cohort")
    return (counts / len(values)).tolist(), float(np.median(clean)) if len(clean) else None


def fit_reference_bins(df, columns, n_bins=10):
    specs = {}
    for column in columns:
        series = df[column]
        clean = series[np.isfinite(series)]
        edges = (
            np.unique(clean.quantile(np.arange(1, n_bins) / n_bins)).tolist()
            if len(clean) else []
        )
        proportions, median = _distribution(series, edges)
        specs[column] = {
            "edges": edges, "proportions": proportions,
            "missing_rate": proportions[-1], "median": median,
        }
    return specs


def _psi_terms(reference_spec, current):
    ref = np.maximum(reference_spec["proportions"], 1e-6)
    cur = np.maximum(current, 1e-6)
    return (cur - ref) * np.log(cur / ref)


def compute_psi(reference_spec, current_series):
    current, _ = _distribution(current_series, reference_spec["edges"])
    return float(_psi_terms(reference_spec, current).sum())


def compute_drift_report(reference_bins, scored_features, scored_probs, score_bins, *,
                         moderate=0.10, significant=0.25):
    def status(psi):
        if psi >= significant:
            return "SIGNIFICANT_SHIFT"
        if psi >= moderate:
            return "MODERATE_SHIFT"
        return "STABLE"

    rows = []
    for feature, spec in reference_bins.items():
        series = scored_features[feature]
        current, median = _distribution(series, spec["edges"])
        terms = _psi_terms(spec, current)
        psi = float(terms.sum())
        rows.append({
            "feature": feature, "psi": psi, "status": status(psi),
            "value_psi": float(terms[:-1].sum()), "missing_psi": float(terms[-1]),
            "ref_missing_pct": 100 * spec["missing_rate"],
            "cur_missing_pct": 100 * current[-1],
            "missing_delta_pct": 100 * (current[-1] - spec["missing_rate"]),
            "ref_median": spec["median"], "cur_median": median,
        })
    score_psi = compute_psi(score_bins, pd.Series(scored_probs))
    return pd.DataFrame(rows), {"score_psi": score_psi, "score_status": status(score_psi)}
