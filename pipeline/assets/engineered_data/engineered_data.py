"""Feature engineering for forklift telemetry.

Reads load-masked CSVs, computes time/delta features, and writes *_forklift_features
files used for training and inference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import FEATURE_CONFIG, PREPROCESSING_CONFIG, FeatureConfig


def engineer_features(df: pd.DataFrame, cfg: FeatureConfig = FEATURE_CONFIG) -> pd.DataFrame:
    """Compute delta/time-based features without mutating the input frame."""
    result = df.copy()
    group_key = "ForkliftID" if "ForkliftID" in result.columns else None

    def _group_diff(column: str) -> pd.Series:
        if column not in result.columns:
            return pd.Series(0.0, index=result.index, dtype="float64")
        if group_key:
            return result.groupby(group_key)[column].diff().fillna(0)
        return result[column].diff().fillna(0)

    if "Timestamp" in result.columns:
        ts_diff = (
            result.groupby(group_key)["Timestamp"].diff().fillna(0)
            if group_key
            else result["Timestamp"].diff().fillna(0)
        )
        result["DeltaTimeSeconds"] = (ts_diff / 1000.0).astype(float)
    else:
        result["DeltaTimeSeconds"] = 0.0

    result["DeltaHeight"] = _group_diff("Height")
    result["DeltaSpeed"] = _group_diff("Speed")

    # Avoid division by zero when timestamps don't advance
    dt = result["DeltaTimeSeconds"].replace(0, np.nan)
    result["HeightChangeRate"] = (result["DeltaHeight"] / dt).replace([np.inf, -np.inf, np.nan], 0.0)

    def _time_since_change(group: pd.DataFrame) -> pd.Series:
        change_flags: pd.Series
        if "LoadChange" in group.columns:
            change_flags = group["LoadChange"].fillna(0)
        elif "Load_Cleaned" in group.columns:
            change_flags = group["Load_Cleaned"].diff().fillna(0)
        elif "OnDuty" in group.columns:
            change_flags = group["OnDuty"].diff().fillna(0)
        else:
            change_flags = pd.Series(0, index=group.index)

        elapsed: list[float] = []
        running = 0.0
        for delta_t, change in zip(group["DeltaTimeSeconds"].fillna(0), change_flags, strict=False):
            if change != 0:
                running = 0.0
            else:
                running += float(delta_t)
            elapsed.append(running)
        return pd.Series(elapsed, index=group.index, dtype="float64")

    time_since = pd.Series(0.0, index=result.index, dtype="float64")
    if group_key:
        for _, group in result.groupby(group_key):
            ts_series = _time_since_change(group)
            time_since.loc[group.index] = ts_series
    else:
        time_since = _time_since_change(result)
    result["TimeSinceLoadChange"] = time_since.astype(float)

    # Ensure all expected feature columns exist
    for col in cfg.feature_columns:
        if col not in result.columns:
            result[col] = 0.0

    return result


def run_feature_engineering(
    load_cleaned_dir: Path | str = PREPROCESSING_CONFIG.output_dir,
    cfg: FeatureConfig = FEATURE_CONFIG,
) -> list[Path]:
    """Generate engineered feature files from load-masked data."""
    load_cleaned_dir = Path(load_cleaned_dir)
    files = sorted(load_cleaned_dir.glob("*_forklift_load_cleaned.csv"))
    if not files:
        raise FileNotFoundError(f"No *_forklift_load_cleaned.csv files found in {load_cleaned_dir}")

    output_dir = cfg.engineered_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_paths: list[Path] = []
    for path in files:
        df = pd.read_csv(path)
        required = {"Height", "Speed", "OnDuty", "Timestamp", "Load_Cleaned"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(sorted(missing))}")

        df = df.sort_values("Timestamp").reset_index(drop=True)

        if "ForkliftID" not in df.columns:
            if "FFFID" in df.columns:
                df["ForkliftID"] = df["FFFID"]
            else:
                df["ForkliftID"] = path.stem.replace("_forklift_load_cleaned", "")

        df["Height_Speed_Interaction"] = df["Height"] * df["Speed"]
        df["Is_Moving"] = (df["Speed"] > cfg.moving_speed_threshold).astype(int)

        engineered_df = engineer_features(df, cfg)

        output_cols = list(cfg.feature_columns) + ["Load_Cleaned", "Timestamp", "ForkliftID"]
        # Preserve any extra context columns if present
        extras = [col for col in ("LoadChange", "Latitude", "Longitude") if col in engineered_df.columns]
        output_df = engineered_df[[c for c in output_cols if c in engineered_df.columns] + extras]

        output_path = output_dir / path.name.replace("_forklift_load_cleaned", "_forklift_features")
        output_df.to_csv(output_path, index=False)
        feature_paths.append(output_path)

    return feature_paths


def main() -> None:
    run_feature_engineering()


if __name__ == "__main__":
    main()
