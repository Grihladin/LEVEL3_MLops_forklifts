from __future__ import annotations

import pandas as pd

# Canonical headers for raw semicolon-delimited files
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


def assign_headers(csv_path) -> pd.DataFrame:
    """Load a raw forklift CSV and attach the canonical headers."""
    return pd.read_csv(
        csv_path,
        sep=";",
        header=None,
        names=COLUMNS,
        engine="python",
    )


def remove_redundant_onduty_zeros(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop consecutive duplicate OnDuty=0 rows; return cleaned frame and count removed."""
    zero_mask = df["OnDuty"].eq(0)
    redundant_zeros = zero_mask & zero_mask.shift(fill_value=False)
    removed = int(redundant_zeros.sum())
    return df[~redundant_zeros].reset_index(drop=True), removed


def coerce_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Convert key numeric fields and drop rows missing Timestamp/Height, then sort."""
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df["Height"] = pd.to_numeric(df["Height"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Height"])
    return df.sort_values("Timestamp").reset_index(drop=True)


def is_forklift(df: pd.DataFrame) -> bool:
    """Return True if any non-zero Height exists."""
    height_values = pd.to_numeric(df["Height"], errors="coerce")
    return height_values.fillna(0).ne(0).any()


def fraction_above_max(df: pd.DataFrame, max_height: float) -> float:
    """Compute fraction of Height readings above max_height."""
    if len(df) == 0:
        return 0.0
    return float((df["Height"] > max_height).mean())
