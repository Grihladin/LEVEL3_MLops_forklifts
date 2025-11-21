"""Clean forklift load sensor data (single run, no parameters).

- Input: height-cleaned forklift files in ``cleaned_data/`` (from clean_data.py).
- Output: load-cleaned files in ``load_cleaned_data/`` with `Load_Cleaned` and updated
  `LoadChange` columns plus a fleet summary CSV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_DIR = Path(__file__).parent / "cleaned_data"
OUTPUT_DIR = Path(__file__).parent / "load_cleaned_data"

MIN_LOAD_DURATION = 30  # seconds
MAX_LOAD_DURATION = 7200  # seconds (2 hours)
MIN_HEIGHT_THRESHOLD = 0.05  # meters
MIN_SPEED_FOR_FILTERING = 15  # km/h


def load_files(data_dir: Path) -> list[Path]:
    files = sorted(p for p in data_dir.glob("*_forklift.csv") if "_broken_height" not in p.name)
    if not files:
        raise FileNotFoundError(f"No *_forklift.csv files found in {data_dir}")
    return files


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


def clean_file(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    required = {"Timestamp", "Height", "Load", "Speed"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} missing required columns: {required - set(df.columns)}")

    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Load_Cleaned"] = df["Load"].copy()

    low_height_mask = (df["Load"] == 1) & (df["Height"] < MIN_HEIGHT_THRESHOLD)
    df.loc[low_height_mask, "Load_Cleaned"] = 0
    height_filtered = int(low_height_mask.sum())

    high_speed_mask = (df["Load"] == 1) & (df["Speed"] > MIN_SPEED_FOR_FILTERING)
    df.loc[high_speed_mask, "Load_Cleaned"] = 0
    speed_filtered = int(high_speed_mask.sum())

    events = detect_load_events(df)
    short_events_filtered = 0
    long_events_filtered = 0

    for event in events:
        if event["duration"] < MIN_LOAD_DURATION:
            df.loc[event["start_idx"] : event["end_idx"], "Load_Cleaned"] = 0
            short_events_filtered += 1
        elif event["duration"] > MAX_LOAD_DURATION:
            df.loc[event["start_idx"] : event["end_idx"], "Load_Cleaned"] = 0
            long_events_filtered += 1

    # recompute events after cleaning so stats reflect final Load_Cleaned
    final_events = detect_load_events(df)

    df["LoadChange"] = df["Load_Cleaned"].diff()

    original_count = len(df)
    original_loaded = int(df["Load"].sum())
    final_loaded = int(df["Load_Cleaned"].sum())
    total_filtered = max(0, original_loaded - final_loaded)
    records_changed = int((df["Load"] != df["Load_Cleaned"]).sum())

    output_path = OUTPUT_DIR / csv_path.name.replace("_forklift", "_forklift_load_cleaned")
    df.to_csv(output_path, index=False)

    return {
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


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = []

    for csv_path in load_files(INPUT_DIR):
        try:
            stats.append(clean_file(csv_path))
            print(f"Cleaned {csv_path.name}")  # noqa: T201
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing {csv_path.name}: {exc}")  # noqa: T201

    if not stats:
        print("No files cleaned.")  # noqa: T201
        return

    stats_df = pd.DataFrame(stats)
    summary_path = OUTPUT_DIR / "cleaning_summary.csv"
    stats_df.to_csv(summary_path, index=False)

    total_original_records = int(stats_df["Original_Records"].sum())
    total_records_changed = int(stats_df["Records_Changed"].sum())
    total_original_loaded = int(stats_df["Original_Loaded"].sum())
    total_final_loaded = int(stats_df["Final_Loaded"].sum())
    total_filtered = int(stats_df["Total_Filtered"].sum())
    total_valid_events = int(stats_df["Valid_Load_Events"].sum())

    print(
        f"Summary: {len(stats)} files, records changed {total_records_changed}/{total_original_records} "
        f"({(total_records_changed/total_original_records*100) if total_original_records else 0:.2f}%), "
        f"filtered loaded {total_filtered}/{total_original_loaded if total_original_loaded else 0} "
        f"({(total_filtered/total_original_loaded*100) if total_original_loaded else 0:.1f}%), "
        f"valid events {total_valid_events}"
    )  # noqa: T201
    print(f"Saved summary to {summary_path}")  # noqa: T201


if __name__ == "__main__":
    run()
