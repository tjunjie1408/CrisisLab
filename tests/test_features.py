import numpy as np
import pandas as pd

from crisislab.features.camels import RATIOS, build_feature_matrix


def test_future_mutation_is_bitwise_invariant_and_lags_respect_gaps():
    panel = pd.DataFrame({
        "CERT": [1, 1, 1, 1, 2],
        "REPDTE": ["2019-06-30", "2019-12-31", "2020-06-30", "2020-09-30", "2020-03-31"],
        **{field: [1., 2., 3., 4., 5.] for field in (
            *RATIOS, "ASSET", "DEP", "EQ", "ORE", "COREDEP", "DEPUNINS",
        )},
    })
    before = build_feature_matrix(panel, "camels_with_trends")
    panel.loc[3, list(panel.columns[2:])] *= 100
    after = build_feature_matrix(panel, "camels_with_trends")
    assert np.array_equal(before.iloc[2].to_numpy().view("uint64"),
                          after.iloc[2].to_numpy().view("uint64"))
    assert np.isnan(before.loc[2, "ROA_d1q"])
    assert before.loc[2, "ROA_d4q"] == 2
    assert before.loc[2, "quarters_observed"] == 2
    assert before.loc[2, "ROA_min4q"] == 2

