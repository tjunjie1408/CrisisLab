import mlflow
import pandas as pd
import pytest

from crisislab.contracts.scoring import get_watchlist_output_schema
from crisislab.models.bundle import CalibratedFailureBundle
from crisislab.pipeline.batch_score import batch_score


def test_batch_replay_and_bundle_integrity(scoring_case, tmp_path):
    snapshot, cfg, manifest = scoring_case
    output = tmp_path / "output"
    single, single_id = batch_score(snapshot, cfg, output, manifest, quarter="2012Q4")
    full, full_id = batch_score(snapshot, cfg, output, manifest)
    assert single_id != full_id
    assert full["quarter"].tolist() == ["2012Q4", "2013Q1"]
    pd.testing.assert_frame_equal(single, full.iloc[:1])
    assert full["realized_failures_4q"].tolist() == [1, 0]
    watch = pd.read_csv(
        output / "watchlist_2012Q4.csv", parse_dates=["REPDTE", "available_date"],
        dtype={"NAME": "string", "STALP": "string"},
    )
    get_watchlist_output_schema().validate(watch)
    client = mlflow.MlflowClient()
    run = client.get_run(full_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["snapshot.financials.sha256"] == "a" * 64
    assert run.data.metrics["quarters_scored"] == 2
    assert run.inputs.dataset_inputs[0].dataset.name == "scored_bank_quarters"
    assert "quarterly_summary.csv" in {item.path for item in client.list_artifacts(full_id)}
    binary = output / "bundle" / "bundle.joblib"
    with binary.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SHA-256"):
        CalibratedFailureBundle.load(output / "bundle")

