from dagster import Definitions, fs_io_manager

from pipeline.config.resources import (
    CleaningResource,
    EvaluationResource,
    FeatureResource,
    PreprocessingResource,
    TrainingResource,
)

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
    resources={
        "io_manager": fs_io_manager,
        "cleaning_settings": CleaningResource(),
        "preprocessing_settings": PreprocessingResource(),
        "feature_settings": FeatureResource(),
        "training_settings": TrainingResource(),
        "evaluation_settings": EvaluationResource(),
    },
)
