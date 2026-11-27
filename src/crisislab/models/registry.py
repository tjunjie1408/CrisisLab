"""Promote the deterministic failure bundle to the MLflow model registry."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import Model, ModelSignature
from mlflow.types import ColSpec, Schema

from crisislab.config import ProjectConfig
from crisislab.models.bundle import CalibratedFailureBundle
from crisislab.pipeline.experiments import load_snapshot
from crisislab.tracking.mlflow_tracker import start_data_centric_run

OUTPUT_COLUMNS = ("raw_prob", "calibrated_prob")


class CalibratedFailurePyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.bundle = CalibratedFailureBundle.load(context.artifacts["bundle_dir"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        missing = set(self.bundle.feature_names).difference(model_input.columns)
        if missing:
            raise ValueError(f"Missing bundle features: {sorted(missing)}")
        return pd.DataFrame(
            dict(zip(OUTPUT_COLUMNS, self.bundle.predict(model_input), strict=True)),
            index=model_input.index,
            dtype=np.float64,
        )


def _load_bundle_artifact(model_uri):
    directory = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))
    flavor = Model.load(str(directory / "MLmodel")).flavors["python_function"]
    bundle_dir = directory / flavor["artifacts"]["bundle_dir"]["path"]
    return CalibratedFailureBundle.load(bundle_dir), bundle_dir


def load_bundle_from_model_uri(model_uri):
    """Load the verified bundle and its fitted PSI references from a trusted model URI."""
    bundle, _ = _load_bundle_artifact(model_uri)
    return bundle


def _assert_predictions(model_uri, bundle, probe):
    observed = mlflow.pyfunc.load_model(model_uri).predict(probe)
    for column, expected in zip(OUTPUT_COLUMNS, bundle.predict(probe), strict=True):
        np.testing.assert_array_equal(expected.view(np.uint64), observed[column].to_numpy().view(
            np.uint64,
        ))


def register_production_model(snapshot, cfg, manifest_path, *, alias=None, bundle_staging_dir=None):
    """Fit, verify, register and promote the failure-only bundle."""
    from crisislab.pipeline.batch_score import _fit_bundle

    if not cfg.tracking.enabled or cfg.pit.include_open_bank_assistance:
        raise ValueError("Promotion requires MLflow tracking and the failure-only target")
    manifest_path = Path(manifest_path)
    for supplied, verified in zip(
        snapshot, load_snapshot(manifest_path.parent, cfg), strict=True,
    ):
        pd.testing.assert_frame_equal(supplied, verified, check_exact=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = {key: entry["sha256"] for key, entry in manifest.items() if "sha256" in entry}
    bundle, _, train_features = _fit_bundle(snapshot, cfg, hashes)
    probe = train_features.head(100).astype(np.float64)
    model_name = cfg.tracking.registered_model_name
    alias = alias if alias is not None else cfg.tracking.production_alias
    signature = ModelSignature(
        inputs=Schema([ColSpec("double", name) for name in bundle.feature_names]),
        outputs=Schema([ColSpec("double", name) for name in OUTPUT_COLUMNS]),
    )
    dependencies = ("mlflow", "cloudpickle", "joblib", "lightgbm", "numpy", "pandas",
                    "scikit-learn", "scipy")
    with TemporaryDirectory() as temporary, start_data_centric_run(
        cfg, experiment_name=cfg.tracking.p3_experiment_name, stage="p3",
        run_name="register-failure-bundle", manifest_path=manifest_path,
    ) as run:
        directory = Path(bundle_staging_dir) if bundle_staging_dir is not None else Path(temporary)
        bundle.save(directory)
        metadata = json.loads((directory / "bundle_metadata.json").read_text(encoding="utf-8"))
        digest = metadata["sha256"]
        tags = {
            "bundle.sha256": digest, "feature_set": cfg.scoring.feature_set,
            "calibration": "platt", "pit_mode": "approximate_pit",
            **{f"snapshot.{key}.sha256": digest for key, digest in hashes.items()},
        }
        mlflow.set_tags(tags)
        model = mlflow.pyfunc.log_model(
            name="failure-bundle", python_model=CalibratedFailurePyFunc(),
            artifacts={"bundle_dir": str(directory)}, signature=signature,
            code_paths=[str(Path(__file__).resolve().parents[1])],
            registered_model_name=model_name,
            pip_requirements=[f"{name}=={version(name)}" for name in dependencies],
        )
        model_version = str(model.registered_model_version)
        client = mlflow.MlflowClient()
        for key, value in tags.items():
            client.set_model_version_tag(model_name, model_version, key, value)
        _assert_predictions(f"models:/{model_name}/{model_version}", bundle, probe)
        client.set_registered_model_alias(model_name, alias, model_version)
        model_uri = f"models:/{model_name}@{alias}"
        _assert_predictions(model_uri, bundle, probe)
        return {
            "run_id": run.info.run_id, "model_name": model_name, "version": model_version,
            "alias": alias, "model_uri": model_uri, "bundle_sha256": digest,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--alias")
    parser.add_argument("--bundle-staging-dir", type=Path)
    args = parser.parse_args()
    cfg = ProjectConfig.from_toml(args.config)
    result = register_production_model(
        load_snapshot(args.raw_dir, cfg), cfg, args.raw_dir / "manifest.json",
        alias=args.alias, bundle_staging_dir=args.bundle_staging_dir,
    )
    print(json.dumps(result, indent=2))
