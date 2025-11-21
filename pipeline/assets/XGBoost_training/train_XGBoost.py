"""Train an XGBoost model to predict forklift load state.

- Input: cleaned load-label files from ``load_cleaned_data/`` (output of load_mask.py).
- Output: model + evaluation plot written to ``artifacts/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "load_cleaned_data"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgboost_load_model.json"
PLOT_PATH = ARTIFACT_DIR / "load_prediction_results.png"
MLFLOW_URI = (ARTIFACT_DIR / "mlruns").as_uri()
EXPERIMENT_NAME = "forklift_load_prediction"


def load_dataset(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*_forklift_load_cleaned.csv"))
    if not files:
        raise FileNotFoundError(f"No *_forklift_load_cleaned.csv files found in {data_dir}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["ForkliftID"] = path.stem.replace("_forklift_load_cleaned", "")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError("Combined dataset is empty after loading files.")
    return combined


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = SPLITS_DIR / "train.csv"
    test_path = SPLITS_DIR / "test.csv"
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)
    raise FileNotFoundError("Train/test splits not found; run preprocessing stage first.")


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = df.copy()
    df["Height_Speed_Interaction"] = df["Height"] * df["Speed"]
    df["Is_Moving"] = (df["Speed"] > 1.0).astype(int)
    feature_cols = ["Height", "Speed", "OnDuty", "Height_Speed_Interaction", "Is_Moving"]
    X = df[feature_cols].fillna(0)
    y = df["Load_Cleaned"].astype(int)
    return X, y, feature_cols


def plot_results(cm, y_test, y_pred_proba, importance_df, accuracy) -> None:
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
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    print(f"✓ Visualization saved: {PLOT_PATH.name}")  # noqa: T201


def main() -> None:
    print("=" * 70)
    print("FORKLIFT LOAD PREDICTION - XGBoost Model")
    print("=" * 70)
    print()

    try:
        train_df, test_df = load_splits()
        print(f"✓ Loaded precomputed splits: {len(train_df):,} train / {len(test_df):,} test")  # noqa: T201
    except FileNotFoundError:
        data = load_dataset(DATA_DIR)
        print(f"✓ Loaded {len(data):,} records from {DATA_DIR}")  # noqa: T201
        train_df, test_df = data, None

    # fixed best params from prior search
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        max_depth=8,
        learning_rate=0.2,
        n_estimators=150,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=5.0,
    )

    print("Training XGBoost model...")  # noqa: T201
    X_train, y_train, features = build_features(train_df)
    model.fit(X_train, y_train)

    X_test = y_test = y_pred = y_pred_proba = None
    accuracy = None
    report = None
    cm = None
    importance_df = None

    if test_df is None:
        print("No test split provided; skipping evaluation.")  # noqa: T201
    else:
        X_test, y_test, _ = build_features(test_df)
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

        plot_results(cm, y_test, y_pred_proba, importance_df, accuracy)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"✓ Model saved: {MODEL_PATH.name}")  # noqa: T201

    # Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="xgboost_load_prediction"):
        mlflow.log_params(
            {
                "max_depth": 8,
                "learning_rate": 0.2,
                "n_estimators": 150,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 5.0,
            }
        )
        if accuracy is not None:
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_dict(report, "classification_report.json")
            if PLOT_PATH.exists():
                mlflow.log_artifact(PLOT_PATH)
        if MODEL_PATH.exists():
            mlflow.log_artifact(MODEL_PATH)

    print("TRAINING COMPLETE")  # noqa: T201


if __name__ == "__main__":
    main()
