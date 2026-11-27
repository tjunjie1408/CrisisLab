import pandas as pd
import pytest
from pandera.errors import SchemaError

from crisislab.contracts.scoring import get_scoring_input_schema, get_watchlist_output_schema
from crisislab.pipeline.batch_score import _fit_bundle, score_quarter


def test_scoring_contracts_and_future_truncation(scoring_case):
    snapshot, cfg, _ = scoring_case
    financials = snapshot[0]
    schema = get_scoring_input_schema()
    with pytest.raises(SchemaError):
        schema.validate(pd.concat([financials, financials.iloc[:1]]))
    with pytest.raises(SchemaError):
        schema.validate(financials.drop(columns="ORE"))
    bundle, _, _ = _fit_bundle(snapshot, cfg, {})
    target = pd.Timestamp("2012-12-31")
    watch, drift, summary = score_quarter(snapshot, target, bundle, cfg)
    changed = financials.copy()
    changed.loc[changed["REPDTE"].gt(target), "ASSET"] = -1.0
    future_failures = snapshot[1].iloc[:2]
    mutated = score_quarter((changed, future_failures, snapshot[2]), target, bundle, cfg)
    pd.testing.assert_frame_equal(mutated[0], watch)
    pd.testing.assert_frame_equal(mutated[1], drift)
    assert mutated[2] == summary
    schema = get_watchlist_output_schema()
    for invalid in [
        pd.concat([watch, watch.iloc[:1]]),
        watch.assign(raw_prob=1.1),
        watch.drop(columns="risk_tier"),
        watch.assign(risk_rank=watch["risk_rank"] + 1),
        watch.iloc[::-1].assign(risk_rank=watch["risk_rank"].to_numpy()),
    ]:
        with pytest.raises(SchemaError):
            schema.validate(invalid)


def test_as_of_closure_with_unknown_insurance_start(scoring_case):
    snapshot, cfg, _ = scoring_case
    financials, failures, _ = snapshot
    bundle, _, _ = _fit_bundle(snapshot, cfg, {})
    institutions = pd.DataFrame({
        "CERT": ["4", "5"],
        "INSDATE": ["12/31/9999", "01/01/2008"],
        "ENDEFYMD": ["03/01/2013", "03/02/2013"],
        "INACTIVE": [1, 1],
        "CHANGEC1": [240, 221],
    })
    watch, _, _ = score_quarter(
        (financials, failures, institutions), pd.Timestamp("2012-12-31"), bundle, cfg,
    )
    assert 4 not in watch["CERT"].to_numpy()
    assert 5 in watch["CERT"].to_numpy()
