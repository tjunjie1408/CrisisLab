import json

import pandas as pd
import pytest

from crisislab.config import ProjectConfig
from crisislab.contracts.scoring import CORE_FIELDS


@pytest.fixture
def scoring_case(tmp_path):
    dates = pd.to_datetime(["2009-06-30", "2011-06-30", "2012-12-31", "2013-03-31"])
    financials = pd.DataFrame([
        {**dict.fromkeys(CORE_FIELDS, float(cert)), "CERT": cert, "REPDTE": date}
        for date in dates for cert in range(1, 8)
        if cert != 1 or date == dates[0]
        if cert != 2 or date <= dates[1]
    ])
    failures = pd.DataFrame({
        "CERT": [1, 2, 3], "FAILDATE": ["2009-12-01", "2011-12-01", "2013-05-30"],
        "RESTYPE": ["FAILURE"] * 3,
    })
    cfg = ProjectConfig()
    cfg.tracking.tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    cfg.tracking.artifact_root = str(tmp_path / "artifacts")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"financials": {"sha256": "a" * 64}}), encoding="utf-8")
    return (financials, failures, pd.DataFrame()), cfg, manifest
