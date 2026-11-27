"""Deterministic estimators and calibration transformations."""

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_baseline(model: str) -> Pipeline | LGBMClassifier:
    if model == "logreg":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                              random_state=42)),
        ])
    if model == "lgbm":
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42,
            deterministic=True, force_col_wise=True, n_jobs=1, verbosity=-1,
        )
    raise ValueError(f"Unknown baseline: {model}")


def _logit(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob, 1e-12, 1 - 1e-12)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_platt(prob, labels):
    return LogisticRegression(random_state=42).fit(_logit(prob), labels)


def platt_predict(calibrator, prob):
    return calibrator.predict_proba(_logit(prob))[:, 1]


def calibrated_probabilities(
    calib_prob: np.ndarray,
    calib_y: np.ndarray,
    score_prob: np.ndarray,
    *,
    train_prior: float,
) -> dict[str, np.ndarray]:
    """Fit on calibration predictions and labels, then transform arbitrary scores."""
    calib_prior = float(np.mean(calib_y))
    if not (0 < train_prior < 1 and 0 < calib_prior < 1):
        raise ValueError("Training and calibration must both contain both classes")
    platt = fit_platt(calib_prob, calib_y)
    platt_prob = platt_predict(platt, score_prob)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(calib_prob, calib_y)
    odds_multiplier = (train_prior / (1 - train_prior)) / (calib_prior / (1 - calib_prior))
    return {
        "uncalibrated": score_prob,
        "platt": platt_prob,
        "isotonic": isotonic.predict(score_prob),
        "prior_adjusted_to_train": (
            odds_multiplier * platt_prob / (1 - platt_prob + odds_multiplier * platt_prob)
        ),
    }


