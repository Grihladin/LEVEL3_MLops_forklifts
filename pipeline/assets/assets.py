from __future__ import annotations

from dagster import MetadataValue, asset

from pipeline.assets.cleaning import cleaning_pipeline
from pipeline.assets.engineered_data import engineered_data
from pipeline.assets.XGBoost_training import train_XGBoost
from pipeline.config.resources import (
    CleaningResource,
    EvaluationResource,
    FeatureResource,
    PreprocessingResource,
    TrainingResource,
)


@asset(name="cleaned_data")
def cleaned_data_asset(
    context, cleaning_settings: CleaningResource, preprocessing_settings: PreprocessingResource
) -> str:
    cleaning_cfg = cleaning_settings.to_config()
    preprocessing_cfg = preprocessing_settings.to_config()

    cleaning_pipeline.run(cleaning_cfg)
    cleaning_pipeline.run_load_cleaning(preprocessing_cfg)

    output_dir = preprocessing_cfg.output_dir
    summary = output_dir / "cleaning_summary.csv"
    metadata = {"output_dir": MetadataValue.path(str(output_dir))}
    if summary.exists():
        metadata["summary_csv"] = MetadataValue.path(str(summary))
    context.log.info(f"Wrote cleaned and load-masked data to {output_dir}")
    context.add_output_metadata(metadata)
    return str(output_dir)


@asset(name="engineered_data", deps=["cleaned_data"])
def engineered_data_asset(
    context, preprocessing_settings: PreprocessingResource, feature_settings: FeatureResource
) -> str:
    preprocessing_cfg = preprocessing_settings.to_config()
    feature_cfg = feature_settings.to_config()

    load_cleaned_dir = preprocessing_cfg.output_dir
    summary = load_cleaned_dir / "cleaning_summary.csv"

    feature_paths = engineered_data.run_feature_engineering(load_cleaned_dir, feature_cfg)
    engineered_dir = feature_cfg.engineered_output_dir

    metadata = {
        "load_cleaned_dir": MetadataValue.path(str(load_cleaned_dir)),
        "engineered_dir": MetadataValue.path(str(engineered_dir)),
    }
    if summary.exists():
        metadata["load_cleaning_summary"] = MetadataValue.path(str(summary))
    if feature_paths:
        metadata["feature_files"] = MetadataValue.md("\n".join([f"- {p.name}" for p in feature_paths]))

    context.log.info(f"Wrote engineered features to {engineered_dir}")
    context.add_output_metadata(metadata)
    return str(engineered_dir)


@asset(name="trained_model", deps=["engineered_data"])
def trained_model_asset(
    context, training_settings: TrainingResource, feature_settings: FeatureResource
) -> str:
    training_cfg = training_settings.to_config()
    feature_cfg = feature_settings.to_config()
    train_XGBoost.main(training_cfg, feature_cfg)
    model_path = training_cfg.model_path
    plot_path = training_cfg.plot_path

    metadata = {
        "model": MetadataValue.path(str(model_path)),
    }
    if plot_path.exists():
        metadata["evaluation_plot"] = MetadataValue.path(str(plot_path))

    context.log.info(f"Trained model saved to {model_path}")
    context.add_output_metadata(metadata)
    return str(model_path)


@asset(name="evaluated_model", deps=["trained_model"])
def evaluated_model_asset(
    context, evaluation_settings: EvaluationResource, feature_settings: FeatureResource
) -> str:
    from pipeline.assets.evaluating import evaluate_model  # local import to avoid eager load during module exec

    evaluation_cfg = evaluation_settings.to_config()
    feature_cfg = feature_settings.to_config()
    metrics = evaluate_model.run(evaluation_cfg, feature_cfg)
    context.log.info(f"Evaluated model with accuracy {metrics['accuracy']:.4f}")
    context.add_output_metadata(
        {
            "accuracy": metrics["accuracy"],
            "confusion_matrix": metrics["confusion_matrix"],
            "report": metrics["report"],
            "confusion_matrix_plot": MetadataValue.path(metrics["confusion_matrix_plot"]),
            "probability_plot": MetadataValue.path(metrics["probability_plot"]),
        }
    )
    return str(evaluation_cfg.metrics_path)
