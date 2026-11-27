"""Declarative project configuration models."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class PITConfig(BaseModel):
    reporting_lag_days: int = Field(default=60, ge=0, le=180)
    horizon_days: int = Field(default=365, gt=0)
    include_open_bank_assistance: bool = False


class PanelConfig(BaseModel):
    start_year: int = Field(default=2000, ge=1992)
    end_year: int = Field(default=2016, ge=1992)
    quarters: list[str] = Field(default_factory=lambda: ["0331", "0630", "0930", "1231"])
    core_camels_fields: list[str] = Field(default_factory=list)


class ReproducibilityConfig(BaseModel):
    random_seed: int = 42
    lightgbm_deterministic: bool = True
    lightgbm_force_col_wise: bool = True


class TrackingConfig(BaseModel):
    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_root: str = "mlartifacts"
    p0_experiment_name: str = "crisislab-p0-data-audit"
    p1_experiment_name: str = "crisislab-p1-data-centric-baselines"


    p2_experiment_name: str = "crisislab-p2-batch-scoring"
    p3_experiment_name: str = "crisislab-p3-model-registry"
    registered_model_name: str = "crisislab-bank-failure-ews"
    production_alias: str = "champion"


class ScoringConfig(BaseModel):
    model: Literal["lgbm"] = "lgbm"
    feature_set: Literal["camels_with_trends"] = "camels_with_trends"
    calibration: Literal["platt"] = "platt"
    top_budget_pct: float = Field(default=2.0, gt=0, le=100)
    tier_critical_threshold: float = 0.05
    tier_high_threshold: float = 0.01
    tier_elevated_threshold: float = 0.0025
    psi_moderate_threshold: float = 0.10
    psi_significant_threshold: float = 0.25
    psi_bins: int = Field(default=10, ge=2)


class ProjectConfig(BaseModel):
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    pit: PITConfig = Field(default_factory=PITConfig)
    panel: PanelConfig = Field(default_factory=PanelConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> ProjectConfig:
        with Path(path).open("rb") as f:
            raw = tomllib.load(f)
        return cls.model_validate(raw)

