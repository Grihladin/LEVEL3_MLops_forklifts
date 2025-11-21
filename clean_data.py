"""End-to-end forklift data cleaner (single run, no parameters).

Steps per raw file in ``data/``:
- Add canonical headers to the semicolon-delimited CSV.
- Drop consecutive redundant OnDuty=0 rows.
- Convert Timestamp/Height to numeric, drop rows missing either, sort by Timestamp.
- Skip non-forklifts (no non-zero Height).
- If >10% of rows have Height above the max, mark the file as broken and write
  `<stem>_forklift_broken_height.csv`.
- Otherwise clean heights (remove rows above the max) and write `<stem>_forklift.csv`.

Outputs go to ``cleaned_data/``; each file is written once.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = [
    "FFFID",
    "Height",
    "Load",
    "OnDuty",
    "Timestamp",
    "Latitude",
    "Longitude",
    "Speed",
]

RAW_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "cleaned_data"

MIN_HEIGHT = 0.0
MAX_HEIGHT = 7.0
BROKEN_HEIGHT_THRESHOLD = 0.10  # fraction of rows above MAX_HEIGHT that marks a broken sensor


def load_raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    return files


def assign_headers(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_path,
        sep=";",
        header=None,
        names=COLUMNS,
        engine="python",
    )


def remove_redundant_onduty_zeros(df: pd.DataFrame) -> pd.DataFrame:
    zero_mask = df["OnDuty"].eq(0)
    redundant_zeros = zero_mask & zero_mask.shift(fill_value=False)
    return df[~redundant_zeros].reset_index(drop=True)


def is_forklift(df: pd.DataFrame) -> bool:
    height_values = pd.to_numeric(df["Height"], errors="coerce")
    return height_values.fillna(0).ne(0).any()


def clean_heights(
    df: pd.DataFrame,
    min_height: float,
    max_height: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {"removed": 0}

    mask = (df["Height"] >= min_height) & (df["Height"] <= max_height)
    stats["removed"] = int((~mask).sum())
    df = df[mask]
    if df.empty:
        raise ValueError("All rows were dropped by height filtering.")

    return df, stats


def process_file(csv_path: Path) -> None:
    df = assign_headers(csv_path)
    df = remove_redundant_onduty_zeros(df)

    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df["Height"] = pd.to_numeric(df["Height"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Height"])
    if df.empty:
        print(f"Skipping {csv_path.name}: no valid rows")  # noqa: T201
        return

    df = df.sort_values("Timestamp").reset_index(drop=True)

    if not is_forklift(df):
        print(f"Skipping {csv_path.name}: not a forklift")  # noqa: T201
        return

    fraction_above_max = (df["Height"] > MAX_HEIGHT).mean()
    if fraction_above_max > BROKEN_HEIGHT_THRESHOLD:
        output_name = f"{csv_path.stem}_forklift_broken_height.csv"
        output_path = OUTPUT_DIR / output_name
        df["Timestamp"] = df["Timestamp"].astype("int64")
        df.to_csv(output_path, index=False)
        print(  # noqa: T201
            f"Flagged broken height: {csv_path.name} -> {output_name} "
            f"({fraction_above_max:.1%} above {MAX_HEIGHT}m)"
        )
        return

    df, stats = clean_heights(df, MIN_HEIGHT, MAX_HEIGHT)
    df["Timestamp"] = df["Timestamp"].astype("int64")
    output_name = f"{csv_path.stem}_forklift.csv"
    output_path = OUTPUT_DIR / output_name
    df.to_csv(output_path, index=False)

    parts = [f"{output_name}: wrote {len(df)} rows"]
    if stats["removed"]:
        parts.append(f"removed {stats['removed']}")
    print("; ".join(parts))  # noqa: T201


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for csv_path in load_raw_files(RAW_DIR):
        process_file(csv_path)


if __name__ == "__main__":
    run()
