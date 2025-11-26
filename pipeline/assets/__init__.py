from dagster import Definitions, fs_io_manager

from pipeline.config.resources import (
    CleaningResource,
    EvaluationResource,
    FeatureResource,
    PreprocessingResource,
    TrainingResource,
)
from pipeline.assets.sensors import materialize_all_assets_job, raw_data_sensor

from .assets import (
    cleaned_data_asset,
    engineered_data_asset,
    trained_model_asset,
    evaluated_model_asset,
)

defs = Definitions(
    assets=[
        cleaned_data_asset,
        engineered_data_asset,
        trained_model_asset,
        evaluated_model_asset,
    ],
    jobs=[materialize_all_assets_job],
    sensors=[raw_data_sensor],
    resources={
        "io_manager": fs_io_manager,
        "cleaning_settings": CleaningResource(),
        "preprocessing_settings": PreprocessingResource(),
        "feature_settings": FeatureResource(),
        "training_settings": TrainingResource(),
        "evaluation_settings": EvaluationResource(),
    },
)
