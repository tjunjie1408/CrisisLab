import numpy as np
import pandas as pd
import pytest

from crisislab.models.baselines import build_baseline, calibrated_probabilities


@pytest.mark.parametrize("kind", ["logreg", "lgbm"])
def test_repeat_fit_has_bitwise_identical_predictions(kind):
    rng = np.random.default_rng(42)
    features = pd.DataFrame(rng.normal(size=(600, 4)))
    labels = (features[0] + features[1] > 0.5).astype(int)
    features.iloc[::11, 0] = np.nan
    train, held_out = features.iloc[:500], features.iloc[500:]
    scores = [
        build_baseline(kind).fit(train, labels.iloc[:500]).predict_proba(held_out)
        for _ in range(2)
    ]
    assert np.array_equal(*scores)
    assert np.ptp(scores[0][:, 1]) > 0


def test_adjustment_uses_training_prior_on_platt_odds():
    calibration = np.array([0.1, 0.2, 0.7, 0.8])
    scores = calibrated_probabilities(
        calibration, np.array([0, 0, 1, 1]), calibration, train_prior=0.2,
    )
    platt = scores["platt"]
    adjusted = scores["prior_adjusted_to_train"]
    np.testing.assert_allclose(adjusted / (1 - adjusted), 0.25 * platt / (1 - platt))

