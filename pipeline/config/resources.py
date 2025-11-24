from __future__ import annotations

from pathlib import Path

from dagster import ConfigurableResource

from .config import (
    CLEANING_CONFIG,
    EVALUATION_CONFIG,
    FEATURE_CONFIG,
    PATHS,
    PREPROCESSING_CONFIG,
    TRAINING_CONFIG,
    CleaningConfig,
    EvaluationConfig,
    FeatureConfig,
    PreprocessingConfig,
    TrainingConfig,
)


class CleaningResource(ConfigurableResource):
    """Dagster-configurable settings for the cleaning stage."""

    raw_dir: str = str(PATHS.raw_data)
    output_dir: str = str(PATHS.cleaned_data)
    min_height: float = CLEANING_CONFIG.min_height
    max_height: float = CLEANING_CONFIG.max_height
    broken_height_threshold: float = CLEANING_CONFIG.broken_height_threshold

    def to_config(self) -> CleaningConfig:
        return CleaningConfig(
            raw_dir=Path(self.raw_dir),
            output_dir=Path(self.output_dir),
            min_height=self.min_height,
            max_height=self.max_height,
            broken_height_threshold=self.broken_height_threshold,
        )


class PreprocessingResource(ConfigurableResource):
    """Dagster-configurable settings for the load-masking stage."""

    input_dir: str = str(PATHS.cleaned_data)
    output_dir: str = str(PATHS.cleaned_data)
    min_load_duration: int = PREPROCESSING_CONFIG.min_load_duration
    max_load_duration: int = PREPROCESSING_CONFIG.max_load_duration
    min_height_threshold: float = PREPROCESSING_CONFIG.min_height_threshold

    def to_config(self) -> PreprocessingConfig:
        return PreprocessingConfig(
            input_dir=Path(self.input_dir),
            output_dir=Path(self.output_dir),
            min_load_duration=self.min_load_duration,
            max_load_duration=self.max_load_duration,
            min_height_threshold=self.min_height_threshold,
        )


class FeatureResource(ConfigurableResource):
    """Dagster-configurable settings for feature engineering."""

    engineered_output_dir: str = str(PATHS.engineered_data)
    moving_speed_threshold: float = FEATURE_CONFIG.moving_speed_threshold
    load_change_rate_threshold: float = FEATURE_CONFIG.load_change_rate_threshold
    feature_columns: list[str] = list(FEATURE_CONFIG.feature_columns)

    def to_config(self) -> FeatureConfig:
        return FeatureConfig(
            engineered_output_dir=Path(self.engineered_output_dir),
            moving_speed_threshold=self.moving_speed_threshold,
            load_change_rate_threshold=self.load_change_rate_threshold,
            feature_columns=tuple(self.feature_columns),
        )


class TrainingResource(ConfigurableResource):
    """Dagster-configurable settings for model training."""

    data_dir: str = str(PATHS.engineered_data)
    splits_dir: str = str(PATHS.splits)
    artifact_dir: str = str(PATHS.artifacts)
    model_path: str = str(TRAINING_CONFIG.model_path)
    plot_path: str = str(TRAINING_CONFIG.plot_path)
    test_size: float = TRAINING_CONFIG.test_size
    mlflow_uri: str = TRAINING_CONFIG.mlflow_uri
    experiment_name: str = TRAINING_CONFIG.experiment_name
    random_state: int = TRAINING_CONFIG.random_state
    max_depth: int = TRAINING_CONFIG.max_depth
    learning_rate: float = TRAINING_CONFIG.learning_rate
    n_estimators: int = TRAINING_CONFIG.n_estimators
    min_child_weight: int = TRAINING_CONFIG.min_child_weight
    subsample: float = TRAINING_CONFIG.subsample
    colsample_bytree: float = TRAINING_CONFIG.colsample_bytree

    def to_config(self) -> TrainingConfig:
        return TrainingConfig(
            data_dir=Path(self.data_dir),
            splits_dir=Path(self.splits_dir),
            artifact_dir=Path(self.artifact_dir),
            model_path=Path(self.model_path),
            plot_path=Path(self.plot_path),
            test_size=self.test_size,
            mlflow_uri=self.mlflow_uri,
            experiment_name=self.experiment_name,
            random_state=self.random_state,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
        )


class EvaluationResource(ConfigurableResource):
    """Dagster-configurable settings for evaluation."""

    splits_dir: str = str(PATHS.splits)
    artifact_dir: str = str(PATHS.artifacts)
    model_path: str = str(EVALUATION_CONFIG.model_path)
    confusion_plot_path: str = str(EVALUATION_CONFIG.confusion_plot_path)
    probability_plot_path: str = str(EVALUATION_CONFIG.probability_plot_path)
    metrics_path: str = str(EVALUATION_CONFIG.metrics_path)

    def to_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            splits_dir=Path(self.splits_dir),
            artifact_dir=Path(self.artifact_dir),
            model_path=Path(self.model_path),
            confusion_plot_path=Path(self.confusion_plot_path),
            probability_plot_path=Path(self.probability_plot_path),
            metrics_path=Path(self.metrics_path),
        )
