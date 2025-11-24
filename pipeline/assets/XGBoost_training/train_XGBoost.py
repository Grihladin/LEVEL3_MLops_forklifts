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

from pipeline.config import FEATURE_CONFIG, TRAINING_CONFIG, FeatureConfig, TrainingConfig


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


def load_splits(splits_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = splits_dir / "train.csv"
    test_path = splits_dir / "test.csv"
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)
    raise FileNotFoundError("Train/test splits not found; run preprocessing stage first.")


def build_features(
    df: pd.DataFrame, feature_cfg: FeatureConfig = FEATURE_CONFIG
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = df.copy()
    df["Height_Speed_Interaction"] = df["Height"] * df["Speed"]
    df["Is_Moving"] = (df["Speed"] > feature_cfg.moving_speed_threshold).astype(int)
    feature_cols = list(feature_cfg.feature_columns)
    X = df[feature_cols].fillna(0)
    y = df["Load_Cleaned"].astype(int)
    return X, y, feature_cols


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


def main(cfg: TrainingConfig = TRAINING_CONFIG, feature_cfg: FeatureConfig = FEATURE_CONFIG) -> None:
    print("=" * 70)
    print("FORKLIFT LOAD PREDICTION - XGBoost Model")
    print("=" * 70)
    print()

    try:
        train_df, test_df = load_splits(cfg.splits_dir)
        print(
            f"✓ Loaded precomputed splits: {len(train_df):,} train / {len(test_df):,} test"
        )  # noqa: T201
    except FileNotFoundError:
        data = load_dataset(cfg.data_dir)
        print(f"✓ Loaded {len(data):,} records from {cfg.data_dir}")  # noqa: T201
        train_df, test_df = data, None

    print("Training XGBoost model...")  # noqa: T201
    X_train, y_train, features = build_features(train_df, feature_cfg)
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    if pos_count == 0:
        print("Warning: no positive samples detected; defaulting scale_pos_weight to 1.0")  # noqa: T201,E501
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = float(neg_count / pos_count)
        print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # fixed best params from prior search with dynamic class weight
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=cfg.random_state,
        use_label_encoder=False,
        eval_metric="logloss",
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        min_child_weight=cfg.min_child_weight,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    X_test = y_test = y_pred = y_pred_proba = None
    accuracy = None
    report = None
    cm = None
    importance_df = None

    if test_df is None:
        print("No test split provided; skipping evaluation.")  # noqa: T201
    else:
        X_test, y_test, _ = build_features(test_df, feature_cfg)
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

        plot_results(cm, y_test, y_pred_proba, importance_df, accuracy, cfg.artifact_dir, cfg.plot_path)

    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(cfg.model_path)
    print(f"✓ Model saved: {cfg.model_path.name}")  # noqa: T201

    # Log to MLflow
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name="xgboost_load_prediction"):
        mlflow.log_params(
            {
                "max_depth": cfg.max_depth,
                "learning_rate": cfg.learning_rate,
                "n_estimators": cfg.n_estimators,
                "min_child_weight": cfg.min_child_weight,
                "subsample": cfg.subsample,
                "colsample_bytree": cfg.colsample_bytree,
                "scale_pos_weight": scale_pos_weight,
            }
        )
        if accuracy is not None:
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_dict(report, "classification_report.json")
            if cfg.plot_path.exists():
                mlflow.log_artifact(cfg.plot_path)
        if cfg.model_path.exists():
            mlflow.log_artifact(cfg.model_path)

    print("TRAINING COMPLETE")  # noqa: T201


if __name__ == "__main__":
    main()
