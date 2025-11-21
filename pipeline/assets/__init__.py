from dagster import Definitions, fs_io_manager

from .assets import (
    cleaned_data_asset,
    load_cleaned_data_asset,
    trained_model_asset,
    evaluated_model_asset,
)

defs = Definitions(
    assets=[
        cleaned_data_asset,
        load_cleaned_data_asset,
        trained_model_asset,
        evaluated_model_asset,
    ],
    resources={"io_manager": fs_io_manager},
)
