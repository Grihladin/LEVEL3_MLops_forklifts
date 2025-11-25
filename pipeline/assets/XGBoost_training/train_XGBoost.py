"""Train an XGBoost model to predict forklift load state.

- Input: feature-engineered files from ``engineered_data/``.
- Splits the dataset into train/test (if not already present) and saves to ``engineered_data/splits/``.
- Output: model + evaluation plot written to ``artifacts/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import mlflow

from pipeline.assets.engineered_data.engineered_data import engineer_features
from pipeline.assets.XGBoost_training.evaluation import evaluate_model, plot_results
from pipeline.config import FEATURE_CONFIG, TRAINING_CONFIG, FeatureConfig, TrainingConfig


def load_dataset(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*_forklift_features.csv"))
    if not files:
        raise FileNotFoundError(
            f"No *_forklift_features.csv files found in {data_dir}; run the engineered_data stage first."
        )

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["ForkliftID"] = path.stem.replace("_forklift_features", "")
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
    raise FileNotFoundError("Train/test splits not found.")


def create_splits(data: pd.DataFrame, cfg: TrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data.empty:
        raise ValueError("Combined dataset is empty after loading files.")

    stratify_col = data["Load_Cleaned"] if data["Load_Cleaned"].nunique() > 1 else None
    train_df, test_df = train_test_split(
        data,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify_col,
    )
    cfg.splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(cfg.splits_dir / "train.csv", index=False)
    test_df.to_csv(cfg.splits_dir / "test.csv", index=False)
    return train_df, test_df


def build_features(
    df: pd.DataFrame, feature_cfg: FeatureConfig = FEATURE_CONFIG
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = df.copy()
    df["Height_Speed_Interaction"] = df["Height"] * df["Speed"]
    df["Is_Moving"] = (df["Speed"] > feature_cfg.moving_speed_threshold).astype(int)
    df = engineer_features(df, feature_cfg)

    feature_cols = list(feature_cfg.feature_columns)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols].fillna(0)
    y = df["Load_Cleaned"].astype(int)
    return X, y, feature_cols


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
        train_df, test_df = create_splits(data, cfg)
        print(
            f"✓ Created stratified splits: {len(train_df):,} train / {len(test_df):,} test "
            f"(saved to {cfg.splits_dir})"
        )  # noqa: T201

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

    evaluation_data: dict[str, object] | None = None

    if test_df is None:
        print("No test split provided; skipping evaluation.")  # noqa: T201
    else:
        X_test, y_test, _ = build_features(test_df, feature_cfg)
        evaluation_data = evaluate_model(model, X_test, y_test)
        plot_results(
            evaluation_data["confusion_matrix"],
            y_test,
            evaluation_data["y_pred_proba"],
            evaluation_data["importance_df"],
            evaluation_data["accuracy"],
            cfg.artifact_dir,
            cfg.plot_path,
        )

    accuracy = evaluation_data["accuracy"] if evaluation_data else None
    report = evaluation_data["report"] if evaluation_data else None
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
            metrics_to_log = {"accuracy": accuracy}
            weighted_avg = report.get("weighted avg", {}) if report else {}
            for metric_name, report_key in (
                ("precision_weighted", "precision"),
                ("recall_weighted", "recall"),
                ("f1_weighted", "f1-score"),
            ):
                value = weighted_avg.get(report_key)
                if value is not None:
                    metrics_to_log[metric_name] = value

            mlflow.log_metrics(metrics_to_log)
            mlflow.log_dict(report, "classification_report.json")
            if cfg.plot_path.exists():
                mlflow.log_artifact(cfg.plot_path)
        if cfg.model_path.exists():
            mlflow.log_artifact(cfg.model_path)

    print("TRAINING COMPLETE")  # noqa: T201


if __name__ == "__main__":
    main()
