from __future__ import annotations

from dagster import MetadataValue, asset

from pipeline.assets.cleaning import cleaning_pipeline
from pipeline.assets.preprocessing import preprocessing
from pipeline.assets.XGBoost_training import train_XGBoost
from pipeline.assets.evaluating import evaluate_model
from pipeline.config import EVALUATION_CONFIG, TRAINING_CONFIG


@asset(name="cleaned_data")
def cleaned_data_asset(context) -> str:
    cleaning_pipeline.run()
    output_dir = cleaning_pipeline.OUTPUT_DIR
    context.log.info(f"Wrote cleaned height data to {output_dir}")
    context.add_output_metadata({"output_dir": MetadataValue.path(str(output_dir))})
    return str(output_dir)


@asset(name="load_cleaned_data", deps=["cleaned_data"])
def load_cleaned_data_asset(context) -> str:
    preprocessing.run()
    output_dir = preprocessing.OUTPUT_DIR
    summary = output_dir / "cleaning_summary.csv"
    metadata = {"output_dir": MetadataValue.path(str(output_dir))}
    if summary.exists():
        metadata["summary_csv"] = MetadataValue.path(str(summary))
    context.log.info(f"Wrote load-cleaned data to {output_dir}")
    context.add_output_metadata(metadata)
    return str(output_dir)


@asset(name="trained_model", deps=["load_cleaned_data"])
def trained_model_asset(context) -> str:
    train_XGBoost.main()
    model_path = TRAINING_CONFIG.model_path
    plot_path = TRAINING_CONFIG.plot_path

    metadata = {
        "model": MetadataValue.path(str(model_path)),
    }
    if plot_path.exists():
        metadata["evaluation_plot"] = MetadataValue.path(str(plot_path))

    context.log.info(f"Trained model saved to {model_path}")
    context.add_output_metadata(metadata)
    return str(model_path)


@asset(name="evaluated_model", deps=["trained_model"])
def evaluated_model_asset(context) -> str:
    metrics = evaluate_model.run()
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
    return str(EVALUATION_CONFIG.metrics_path)
