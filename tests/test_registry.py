import hashlib
import json

import mlflow
import numpy as np
import pandas as pd

from crisislab.models.bundle import CalibratedFailureBundle
from crisislab.models.registry import load_bundle_from_model_uri, register_production_model
from crisislab.pipeline.batch_score import batch_score


def test_registry_promotion_and_batch_replay(scoring_case, tmp_path, monkeypatch):
    snapshot, cfg, manifest = scoring_case
    entries = {}
    for name, frame in zip(
        ("financials_2000_2016", "failures", "institutions"), snapshot, strict=True,
    ):
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path)
        entries[name] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    standalone = tmp_path / "standalone"
    registry = tmp_path / "registry"
    batch_score(snapshot, cfg, standalone, manifest, quarter="2012Q4")
    metadata = register_production_model(snapshot, cfg, manifest)
    uri = metadata["model_uri"]
    client = mlflow.MlflowClient()
    promoted = client.get_model_version_by_alias(metadata["model_name"], "champion")
    assert str(promoted.version) == metadata["version"] == "1"
    assert promoted.tags["bundle.sha256"] == metadata["bundle_sha256"]
    assert client.get_run(metadata["run_id"]).info.status == "FINISHED"

    bundle = CalibratedFailureBundle.load(standalone / "bundle")
    probe = pd.DataFrame(1.0, columns=bundle.feature_names, index=[0, 1])
    result = mlflow.pyfunc.load_model(uri).predict(probe)
    for column, expected in zip(result.columns, bundle.predict(probe), strict=True):
        np.testing.assert_array_equal(
            result[column].to_numpy().view("uint64"), expected.view("uint64"),
        )
    for model_uri in (uri, f"runs:/{metadata['run_id']}/failure-bundle"):
        assert load_bundle_from_model_uri(model_uri).reference_feature_bins == (
            bundle.reference_feature_bins
        )

    def reject_refit(*args, **kwargs):
        raise AssertionError("Registry scoring attempted to fit a model")

    monkeypatch.setattr(CalibratedFailureBundle, "fit", reject_refit)
    _, run_id = batch_score(snapshot, cfg, registry, manifest, quarter="2012Q4", model_uri=uri)
    watchlist = "watchlist_2012Q4.csv"
    assert (standalone / watchlist).read_bytes() == (registry / watchlist).read_bytes()
    drift = pd.read_csv(registry / "drift_2012Q4.csv")
    np.testing.assert_allclose(drift.value_psi + drift.missing_psi, drift.psi)
    assert client.get_run(run_id).data.params["bundle.sha256"] == metadata["bundle_sha256"]
