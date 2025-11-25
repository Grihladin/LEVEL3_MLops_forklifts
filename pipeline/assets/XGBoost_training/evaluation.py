"""Evaluation and visualization utilities for the XGBoost forklift model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(
    model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, object]:
    """Run classification evaluation and return metrics plus helper artifacts."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Unloaded", "Loaded"], output_dict=True)

    print(f"Accuracy: {accuracy*100:.2f}%")  # noqa: T201
    print(classification_report(y_test, y_pred, target_names=["Unloaded", "Loaded"]))  # noqa: T201

    cm = confusion_matrix(y_test, y_pred)
    importance_dict = model.get_booster().get_score(importance_type="weight")
    importance_df = (
        pd.DataFrame([{"feature": k, "importance": v} for k, v in importance_dict.items()])
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("Feature importance (weight):")  # noqa: T201
    print(importance_df.to_string(index=False))  # noqa: T201

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "y_pred_proba": y_pred_proba,
        "importance_df": importance_df,
    }


def plot_results(
    cm,
    y_test,
    y_pred_proba,
    importance_df,
    accuracy,
    artifact_dir: Path,
    plot_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_xticklabels(["Unloaded", "Loaded"])
    ax1.set_yticklabels(["Unloaded", "Loaded"])

    ax2 = axes[0, 1]
    top_imp = importance_df.head(5)
    ax2.barh(top_imp["feature"], top_imp["importance"], color="#3498db")
    ax2.set_title("Top 5 Feature Importance", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Importance Score")
    ax2.invert_yaxis()

    ax3 = axes[1, 0]
    ax3.hist(
        [y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]],
        bins=50,
        label=["Actual Unloaded", "Actual Loaded"],
        color=["#3498db", "#e74c3c"],
        alpha=0.7,
    )
    ax3.axvline(x=0.5, color="black", linestyle="--", linewidth=2, label="Decision Threshold")
    ax3.set_title("Prediction Probability Distribution", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Predicted Probability (Loaded)")
    ax3.set_ylabel("Frequency")
    ax3.legend()

    ax4 = axes[1, 1]
    ax4.axis("off")
    metrics_text = f"""
MODEL PERFORMANCE SUMMARY

Accuracy: {accuracy*100:.2f}%

Test Set Size: {len(y_test):,}
  Unloaded: {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)
  Loaded: {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)
"""
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, family="monospace", verticalalignment="center")

    plt.tight_layout()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✓ Visualization saved: {plot_path.name}")  # noqa: T201
