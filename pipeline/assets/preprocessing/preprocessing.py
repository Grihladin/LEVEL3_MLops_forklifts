"""Stage 2 preprocessing: apply load mask to cleaned forklifts and prepare splits.

- Reads height-cleaned forklifts from ``cleaned_data/`` (skips broken height files).
- Applies height/speed filters and short/long event removal to produce ``Load_Cleaned``.
- Writes per-file outputs to ``load_cleaned_data/`` and a summary CSV.
- Concatenates all cleaned frames and writes stratified train/test splits to ``load_cleaned_data/splits/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.config import PREPROCESSING_CONFIG, PreprocessingConfig


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


def clean_file(csv_path: Path, cfg: PreprocessingConfig = PREPROCESSING_CONFIG) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    required = {"Timestamp", "Height", "Load", "Speed"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} missing required columns: {required - set(df.columns)}")

    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Load_Cleaned"] = df["Load"].copy()

    low_height_mask = (df["Load"] == 1) & (df["Height"] < cfg.min_height_threshold)
    df.loc[low_height_mask, "Load_Cleaned"] = 0
    height_filtered = int(low_height_mask.sum())

    high_speed_mask = (df["Load"] == 1) & (df["Speed"] > cfg.min_speed_for_filtering)
    df.loc[high_speed_mask, "Load_Cleaned"] = 0
    speed_filtered = int(high_speed_mask.sum())

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
        "Speed_Filtered": speed_filtered,
        "Short_Events_Filtered": short_events_filtered,
        "Long_Events_Filtered": long_events_filtered,
        "Valid_Load_Events": len(final_events),
        "OutputFile": output_path.name,
    }
    return stats, df


def run(cfg: PreprocessingConfig = PREPROCESSING_CONFIG) -> None:
    files = sorted(
        p for p in cfg.input_dir.glob("*_forklift.csv") if "_broken_height" not in p.name
    )
    if not files:
        raise FileNotFoundError(f"No *_forklift.csv files found in {cfg.input_dir}")

    stats = []
    cleaned_frames: list[pd.DataFrame] = []
    for csv_path in files:
        file_stats, cleaned_df = clean_file(csv_path, cfg)
        stats.append(file_stats)
        cleaned_frames.append(cleaned_df)
        print(f"Cleaned {csv_path.name}")  # noqa: T201

    stats_df = pd.DataFrame(stats)
    summary_path = cfg.output_dir / "cleaning_summary.csv"
    stats_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")  # noqa: T201

    if cleaned_frames:
        combined = pd.concat(cleaned_frames, ignore_index=True)
        train_df, test_df = train_test_split(
            combined,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=combined["Load_Cleaned"],
        )
        (cfg.output_dir / "splits").mkdir(parents=True, exist_ok=True)
        train_df.to_csv(cfg.output_dir / "splits" / "train.csv", index=False)
        test_df.to_csv(cfg.output_dir / "splits" / "test.csv", index=False)
        print(f"Wrote train/test splits to {cfg.output_dir / 'splits'}")  # noqa: T201


if __name__ == "__main__":
    run()
