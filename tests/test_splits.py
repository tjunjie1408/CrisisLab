import pandas as pd

from crisislab.contracts.pit import assert_three_way_no_leakage, split_train_calib_test


def test_three_way_embargo_and_primary_test_retains_exits():
    available = pd.Series(pd.to_datetime([
        "2009-11-29", "2010-01-01", "2010-03-01", "2010-11-29", "2011-03-01",
        "2012-01-02", "2012-02-29", "2012-11-29", "2013-03-01", "2013-03-01",
    ]))
    panel = pd.DataFrame({
        "CERT": range(1, 11),
        "REPDTE": available - pd.Timedelta(days=60),
        "available_date": available,
        "label_mature_date": available + pd.Timedelta(days=365),
        "eligible_at_scoring_time": [True] * 9 + [False],
        "fail_date": pd.NaT,
        "non_failure_exit_in_horizon": [1] + [0] * 7 + [1, 0],
    })
    parts = split_train_calib_test(panel)
    assert_three_way_no_leakage(*parts)
    assert [part.index.tolist() for part in parts] == [[0, 1], [4, 5], [8]]
    censored = split_train_calib_test(panel, censor_train_exits=True)
    assert censored[0].index.tolist() == [1]
    pd.testing.assert_frame_equal(parts[-1], censored[-1])

