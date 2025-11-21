from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from pipeline.assets.XGBoost_training.train_XGBoost import build_features
from pipeline.config import EVALUATION_CONFIG, FEATURE_CONFIG, EvaluationConfig, FeatureConfig


def load_test_split(splits_dir: Path) -> pd.DataFrame:
    test_path = splits_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError("Test split not found; run preprocessing stage first.")
    return pd.read_csv(test_path)


def load_model(model_path: Path) -> xgb.XGBClassifier:
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found; run training stage first.")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model


def plot_confusion(cm, artifact_dir: Path, output_path: Path) -> Path:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0.5, 1.5], ["Unloaded", "Loaded"])
    plt.yticks([0.5, 1.5], ["Unloaded", "Loaded"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_probabilities(y_test, y_pred_proba, artifact_dir: Path, output_path: Path) -> Path:
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
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def run(cfg: EvaluationConfig = EVALUATION_CONFIG, feature_cfg: FeatureConfig = FEATURE_CONFIG) -> dict:
    test_df = load_test_split(cfg.splits_dir)
    X_test, y_test, _ = build_features(test_df, feature_cfg)

    model = load_model(cfg.model_path)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Unloaded", "Loaded"], output_dict=True)

    confusion_path = plot_confusion(cm, cfg.artifact_dir, cfg.confusion_plot_path)
    prob_path = plot_probabilities(y_test, y_pred_proba, cfg.artifact_dir, cfg.probability_plot_path)

    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "report": report,
        "confusion_matrix_plot": str(confusion_path),
        "probability_plot": str(prob_path),
    }
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    with cfg.metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    run()
