"""Train-only baseline and PSI references with calibration-only Platt fitting."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from crisislab.models.baselines import build_baseline, fit_platt, platt_predict
from crisislab.monitoring.drift import fit_reference_bins


@dataclass
class CalibratedFailureBundle:
    model: object
    calibrator: object
    feature_names: list[str]
    train_prior: float
    calib_prior: float
    calib_top2_threshold: float
    reference_feature_bins: dict
    reference_score_bins: dict
    snapshot_hashes: dict

    @classmethod
    def fit(cls, train, train_y, calib, calib_y, snapshot_hashes, *, n_bins=10):
        model = build_baseline("lgbm").fit(train, train_y)
        calibrator = fit_platt(model.predict_proba(calib)[:, 1], calib_y)
        calib_probs = platt_predict(calibrator, model.predict_proba(calib)[:, 1])
        train_probs = platt_predict(calibrator, model.predict_proba(train)[:, 1])
        monitored = train.columns
        return cls(
            model, calibrator, train.columns.tolist(), float(np.mean(train_y)),
            float(np.mean(calib_y)),
            float(np.sort(calib_probs)[-math.ceil(len(calib_probs) * 0.02)]),
            fit_reference_bins(train, monitored, n_bins),
            fit_reference_bins(pd.DataFrame({"score": train_probs}), ["score"], n_bins)["score"],
            snapshot_hashes,
        )

    def predict(self, features):
        raw = self.model.predict_proba(features[self.feature_names])[:, 1]
        return raw, platt_predict(self.calibrator, raw)

    def save(self, bundle_dir):
        directory = Path(bundle_dir)
        directory.mkdir(parents=True, exist_ok=True)
        binary = directory / "bundle.joblib"
        joblib.dump(self, binary)
        metadata = {
            key: value for key, value in vars(self).items()
            if key not in {"model", "calibrator"}
        }
        metadata.update(sha256=_digest(binary), score_reference_cohort="Train")
        (directory / "bundle_metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )

    @classmethod
    def load(cls, bundle_dir):
        """Load a trusted bundle after verifying its binary digest."""
        directory = Path(bundle_dir)
        metadata = json.loads((directory / "bundle_metadata.json").read_text(encoding="utf-8"))
        binary = directory / "bundle.joblib"
        if _digest(binary) != metadata["sha256"]:
            raise ValueError("Bundle SHA-256 mismatch")
        return joblib.load(binary)


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
