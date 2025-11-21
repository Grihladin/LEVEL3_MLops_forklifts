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
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "load_cleaned_data"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgboost_load_model.json"
PLOT_PATH = ARTIFACT_DIR / "load_prediction_results.png"


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

    data = load_dataset(DATA_DIR)
    print(f"✓ Loaded {len(data):,} records from {DATA_DIR}")  # noqa: T201

    X, y, feature_cols = build_features(data)
    print(f"Features: {feature_cols}")  # noqa: T201
    print(f"Loaded samples: {y.sum():,} ({y.mean()*100:.2f}%)")  # noqa: T201
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
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

    print("TRAINING COMPLETE")  # noqa: T201


if __name__ == "__main__":
    main()
