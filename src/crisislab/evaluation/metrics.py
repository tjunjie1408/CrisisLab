"""Ranking, probability quality, and historical early-warning diagnostics."""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def evaluate_scores(panel: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    """Evaluate a pooled row budget, breaking score ties by date and CERT."""
    ranked = panel.assign(probability=probabilities).sort_values(
        ["probability", "available_date", "CERT"], ascending=[False, True, True]
    )
    y = ranked["failed_in_horizon"]
    p = ranked["probability"]
    positives = int(y.sum())
    negatives = len(y) - positives
    rate = y.mean()
    bins = np.minimum((p * 10).astype(int), 9)
    calibration = ranked.groupby(bins).agg(
        actual=("failed_in_horizon", "mean"), predicted=("probability", "mean"),
        count=("probability", "size"),
    )
    metrics = {
        "rows": len(y),
        "positive_rows": positives,
        "pr_auc": average_precision_score(y, p) if positives else np.nan,
        "roc_auc": roc_auc_score(y, p) if positives and negatives else np.nan,
        "brier_score": brier_score_loss(y, p) if len(y) else np.nan,
        "log_loss": log_loss(y, p, labels=[0, 1]) if len(y) else np.nan,
        "ece_10bin": (
            (calibration["actual"] - calibration["predicted"]).abs() @ calibration["count"]
        ) / len(y) if len(y) else np.nan,
        "mean_pred_prob": p.mean(),
        "base_rate": rate,
        "pred_to_actual_ratio": p.mean() / rate if positives else np.nan,
    }
    for pct in (1, 2, 5):
        alerts = ranked.head(math.ceil(len(y) * pct / 100))
        true_alerts = alerts.loc[alerts["failed_in_horizon"].eq(1)]
        tp = len(true_alerts)
        metrics[f"recall_at_top_{pct}pct"] = tp / positives if positives else np.nan
        if pct == 2:
            event_count = ranked.loc[y.eq(1), "CERT"].nunique()
            first = true_alerts.sort_values("available_date").drop_duplicates("CERT")
            lead = (first["fail_date"] - first["available_date"]).dt.days
            metrics.update(
                precision_at_top_2pct=tp / len(alerts) if len(alerts) else np.nan,
                false_positive_rate_at_top_2pct=(len(alerts) - tp) / negatives
                if negatives else np.nan,
                failure_events=event_count,
                event_recall_at_top_2pct=len(first) / event_count if event_count else np.nan,
                median_lead_time_days_at_top_2pct=lead.median() if len(first) else np.nan,
            )
    return metrics


def slice_diagnostics(
    test: pd.DataFrame, probabilities: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute the same budget within each slice using fixed model predictions."""
    year = test["available_date"].dt.year
    asset = pd.to_numeric(test["ASSET"], errors="coerce")
    tiers = {
        "<$100M": asset.lt(100_000),
        "$100M–<$1B": asset.ge(100_000) & asset.lt(1_000_000),
        "$1B–$10B": asset.ge(1_000_000) & asset.le(10_000_000),
        ">$10B": asset.gt(10_000_000),
    }

    def table(masks: dict) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {name: evaluate_scores(test.loc[mask], probabilities[mask.to_numpy()])
             for name, mask in masks.items()}, orient="index"
        )

    return (
        table({yr: year.eq(yr) for yr in sorted(year.unique())}),
        table(tiers),
        table({
            "Primary uncensored": pd.Series(True, index=test.index),
            "Offline non-failure-exit censored": test["non_failure_exit_in_horizon"].eq(0),
        }),
    )

