from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow

from pipeline.assets.XGBoost_training.train_XGBoost import build_features
from pipeline.config import EVALUATION_CONFIG, TRAINING_CONFIG

CFG = EVALUATION_CONFIG
SPLITS_DIR = CFG.splits_dir
ARTIFACT_DIR = CFG.artifact_dir
MODEL_PATH = CFG.model_path
CONFUSION_PLOT_PATH = CFG.confusion_plot_path
PROB_PLOT_PATH = CFG.probability_plot_path
METRICS_PATH = CFG.metrics_path
MLFLOW_URI = CFG.mlflow_uri
EXPERIMENT_NAME = CFG.experiment_name


def load_test_split() -> pd.DataFrame:
    test_path = SPLITS_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError("Test split not found; run preprocessing stage first.")
    return pd.read_csv(test_path)


def load_model() -> xgb.XGBClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found; run training stage first.")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def plot_confusion(cm) -> Path:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0.5, 1.5], ["Unloaded", "Loaded"])
    plt.yticks([0.5, 1.5], ["Unloaded", "Loaded"])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(CONFUSION_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    return CONFUSION_PLOT_PATH


def plot_probabilities(y_test, y_pred_proba) -> Path:
    plt.figure(figsize=(7, 5))
    plt.hist(
        [y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]],
        bins=50,
        label=["Actual Unloaded", "Actual Loaded"],
        color=["#3498db", "#e74c3c"],
        alpha=0.7,
    )
    plt.axvline(x=0.5, color="black", linestyle="--", linewidth=2, label="Decision Threshold")
    plt.title("Prediction Probability Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Probability (Loaded)")
    plt.ylabel("Frequency")
    plt.legend()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PROB_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    return PROB_PLOT_PATH


def run() -> dict:
    test_df = load_test_split()
    X_test, y_test, _ = build_features(test_df)

    model = load_model()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Unloaded", "Loaded"], output_dict=True)

    confusion_path = plot_confusion(cm)
    prob_path = plot_probabilities(y_test, y_pred_proba)

    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "report": report,
        "confusion_matrix_plot": str(confusion_path),
        "probability_plot": str(prob_path),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)

    # Log to MLflow as a separate evaluation run
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="evaluation", description="Evaluation of trained model"):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_dict(report, "classification_report.json")
        mlflow.log_artifact(CONFUSION_PLOT_PATH)
        mlflow.log_artifact(PROB_PLOT_PATH)
        mlflow.log_artifact(METRICS_PATH)

    return metrics


if __name__ == "__main__":
    run()
