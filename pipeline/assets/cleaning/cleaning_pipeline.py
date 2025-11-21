"""Stage 1 cleaning pipeline: label, filter, and write height-cleaned forklift files.

- Reads raw semicolon-delimited CSVs from ``data/``.
- Adds canonical headers and removes redundant OnDuty=0 rows.
- Drops rows with invalid/missing Timestamp/Height and sorts by Timestamp.
- Skips non-forklifts (no non-zero Height).
- Flags broken height sensors (>10% above MAX_HEIGHT) and saves as *_broken_height.
- Otherwise drops rows outside [MIN_HEIGHT, MAX_HEIGHT] and saves to ``cleaned_data/``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.config import CLEANING_CONFIG, CleaningConfig

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


if __name__ == "__main__":
    run()
