"""Cleaning pipeline for forklift telemetry.

Stage 1 (height cleaning):
- Reads raw semicolon-delimited CSVs from ``data/``.
- Adds canonical headers and removes redundant OnDuty=0 rows.
- Drops rows with invalid/missing Timestamp/Height and sorts by Timestamp.
- Skips non-forklifts (no non-zero Height).
- Flags broken height sensors (>10% above MAX_HEIGHT) and saves as *_broken_height.
- Otherwise drops rows outside [MIN_HEIGHT, MAX_HEIGHT] and saves to ``cleaned_data/``.

Stage 2 (load masking):
- Reads height-cleaned forklifts from ``cleaned_data/`` (skips broken height files).
- Applies height filters and short/long event removal to produce ``Load_Cleaned``.
- Writes per-file outputs to ``cleaned_data/`` (with ``_load_cleaned`` suffix) and a summary CSV.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.config import CLEANING_CONFIG, PREPROCESSING_CONFIG, CleaningConfig, PreprocessingConfig

from . import cleaning_helpers as helpers


def clean_file(csv_path: Path, cfg: CleaningConfig = CLEANING_CONFIG) -> tuple[str | None, dict]:
    df = helpers.assign_headers(csv_path)
    df, removed_onduty = helpers.remove_redundant_onduty_zeros(df)
    df = helpers.coerce_and_sort(df)

    if df.empty:
        return None, {"skipped": "no_valid_rows"}

    if not helpers.is_forklift(df):
        return None, {"skipped": "not_forklift"}

    frac_above = helpers.fraction_above_max(df, cfg.max_height)
    if frac_above > cfg.broken_height_threshold:
        output_name = f"{csv_path.stem}_forklift_broken_height.csv"
        output_path = cfg.output_dir / output_name
        df["Timestamp"] = df["Timestamp"].astype("int64")
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_name, {
            "removed_onduty": removed_onduty,
            "broken_height_fraction": frac_above,
            "status": "broken_height",
        }

    mask = (df["Height"] >= cfg.min_height) & (df["Height"] <= cfg.max_height)
    removed_height = int((~mask).sum())
    df = df[mask]
    if df.empty:
        return None, {"skipped": "all_height_out_of_range"}

    df["Timestamp"] = df["Timestamp"].astype("int64")
    output_name = f"{csv_path.stem}_forklift.csv"
    output_path = cfg.output_dir / output_name
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return output_name, {
        "removed_onduty": removed_onduty,
        "removed_height": removed_height,
        "broken_height_fraction": frac_above,
        "status": "cleaned",
    }


def detect_load_events(df: pd.DataFrame) -> list[dict]:
    events = []
    start_idx: int | None = None

    for idx in range(len(df)):
        current = df.at[idx, "Load_Cleaned"]
        prev = df.at[idx - 1, "Load_Cleaned"] if idx > 0 else 0
        if current == 1 and prev == 0:
            start_idx = idx
        elif current == 0 and prev == 1 and start_idx is not None:
            end_idx = idx - 1
            duration = (df.at[end_idx, "Timestamp"] - df.at[start_idx, "Timestamp"]) / 1000
            events.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "duration": duration,
                    "start_time": df.at[start_idx, "Timestamp"],
                    "end_time": df.at[end_idx, "Timestamp"],
                }
            )
            start_idx = None

    if start_idx is not None:
        end_idx = len(df) - 1
        duration = (df.at[end_idx, "Timestamp"] - df.at[start_idx, "Timestamp"]) / 1000
        events.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "duration": duration,
                "start_time": df.at[start_idx, "Timestamp"],
                "end_time": df.at[end_idx, "Timestamp"],
            }
        )

    return events


def apply_load_mask(csv_path: Path, cfg: PreprocessingConfig = PREPROCESSING_CONFIG) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    required = {"Timestamp", "Height", "Load", "Speed"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} missing required columns: {required - set(df.columns)}")

    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Load_Cleaned"] = df["Load"].copy()

    low_height_mask = (df["Load"] == 1) & (df["Height"] < cfg.min_height_threshold)
    df.loc[low_height_mask, "Load_Cleaned"] = 0
    height_filtered = int(low_height_mask.sum())

    events = detect_load_events(df)
    short_events_filtered = 0
    long_events_filtered = 0

    for event in events:
        if event["duration"] < cfg.min_load_duration:
            df.loc[event["start_idx"] : event["end_idx"], "Load_Cleaned"] = 0
            short_events_filtered += 1
        elif event["duration"] > cfg.max_load_duration:
            df.loc[event["start_idx"] : event["end_idx"], "Load_Cleaned"] = 0
            long_events_filtered += 1

    final_events = detect_load_events(df)
    df["LoadChange"] = df["Load_Cleaned"].diff()

    original_count = len(df)
    original_loaded = int(df["Load"].sum())
    final_loaded = int(df["Load_Cleaned"].sum())
    total_filtered = max(0, original_loaded - final_loaded)
    records_changed = int((df["Load"] != df["Load_Cleaned"]).sum())

    output_path = cfg.output_dir / csv_path.name.replace("_forklift", "_forklift_load_cleaned")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    stats = {
        "ForkliftID": csv_path.stem.replace("_forklift", ""),
        "Original_Records": original_count,
        "Original_Loaded": original_loaded,
        "Final_Loaded": final_loaded,
        "Total_Filtered": total_filtered,
        "Filter_Percentage": (total_filtered / original_loaded * 100) if original_loaded else 0,
        "Records_Changed": records_changed,
        "Percentage_Changed": (records_changed / original_count * 100) if original_count else 0,
        "Height_Filtered": height_filtered,
        "Short_Events_Filtered": short_events_filtered,
        "Long_Events_Filtered": long_events_filtered,
        "Valid_Load_Events": len(final_events),
        "OutputFile": output_path.name,
    }
    return stats, df


def run(cfg: CleaningConfig = CLEANING_CONFIG) -> None:
    raw_files = sorted(cfg.raw_dir.glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No CSV files found in {cfg.raw_dir}")

    for csv_path in raw_files:
        name, info = clean_file(csv_path, cfg)
        if name:
            print(f"{csv_path.name} -> {name} ({info.get('status')})")  # noqa: T201
        else:
            print(f"Skipped {csv_path.name}: {info.get('skipped')}")  # noqa: T201


def run_load_cleaning(cfg: PreprocessingConfig = PREPROCESSING_CONFIG) -> None:
    files = sorted(
        p for p in cfg.input_dir.glob("*_forklift.csv") if "_broken_height" not in p.name
    )
    if not files:
        raise FileNotFoundError(f"No *_forklift.csv files found in {cfg.input_dir}")

    stats = []
    for csv_path in files:
        file_stats, _ = apply_load_mask(csv_path, cfg)
        stats.append(file_stats)
        print(f"Load-masked {csv_path.name}")  # noqa: T201

    stats_df = pd.DataFrame(stats)
    summary_path = cfg.output_dir / "cleaning_summary.csv"
    stats_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")  # noqa: T201


if __name__ == "__main__":
    run()
