"""Centralized configuration for paths, thresholds, and model settings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Project paths (repo root)
ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT_DIR
    raw_data: Path = ROOT_DIR / "data"
    cleaned_data: Path = ROOT_DIR / "cleaned_data"
    load_cleaned_data: Path = ROOT_DIR / "load_cleaned_data"
    artifacts: Path = ROOT_DIR / "artifacts"
    mlruns: Path = artifacts / "mlruns"
    splits: Path = load_cleaned_data / "splits"


PATHS = Paths()


@dataclass(frozen=True)
class CleaningConfig:
    raw_dir: Path = PATHS.raw_data
    output_dir: Path = PATHS.cleaned_data
    min_height: float = 0.0
    max_height: float = 7.0
    broken_height_threshold: float = 0.10


@dataclass(frozen=True)
class PreprocessingConfig:
    input_dir: Path = PATHS.cleaned_data
    output_dir: Path = PATHS.load_cleaned_data
    min_load_duration: int = 30  # seconds
    max_load_duration: int = 7200  # seconds
    min_height_threshold: float = 0.05  # meters
    min_speed_for_filtering: float = 15  # km/h
    test_size: float = 0.2
    random_state: int = 42


@dataclass(frozen=True)
class FeatureConfig:
    moving_speed_threshold: float = 1.0  # km/h for Is_Moving flag
    feature_columns: tuple[str, ...] = (
        "Height",
        "Speed",
        "OnDuty",
        "Height_Speed_Interaction",
        "Is_Moving",
    )


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path = PATHS.load_cleaned_data
    splits_dir: Path = PATHS.splits
    artifact_dir: Path = PATHS.artifacts
    model_path: Path = artifact_dir / "xgboost_load_model.json"
    plot_path: Path = artifact_dir / "load_prediction_results.png"
    mlflow_uri: str = PATHS.mlruns.as_uri()
    experiment_name: str = "forklift_load_prediction"
    random_state: int = 42
    max_depth: int = 8
    learning_rate: float = 0.2
    n_estimators: int = 150
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: float = 5.0


@dataclass(frozen=True)
class EvaluationConfig:
    splits_dir: Path = PATHS.splits
    artifact_dir: Path = PATHS.artifacts
    model_path: Path = PATHS.artifacts / "xgboost_load_model.json"
    confusion_plot_path: Path = PATHS.artifacts / "confusion_matrix.png"
    probability_plot_path: Path = PATHS.artifacts / "probability_distribution.png"
    metrics_path: Path = PATHS.artifacts / "evaluation_metrics.json"
    mlflow_uri: str = PATHS.mlruns.as_uri()
    experiment_name: str = "forklift_load_prediction"


CLEANING_CONFIG = CleaningConfig()
PREPROCESSING_CONFIG = PreprocessingConfig()
FEATURE_CONFIG = FeatureConfig()
TRAINING_CONFIG = TrainingConfig()
EVALUATION_CONFIG = EvaluationConfig()


__all__ = [
    "PATHS",
    "CLEANING_CONFIG",
    "PREPROCESSING_CONFIG",
    "FEATURE_CONFIG",
    "TRAINING_CONFIG",
    "EVALUATION_CONFIG",
]
