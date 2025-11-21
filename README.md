# Forklift Data Processing

This repo cleans raw forklift telemetry in one pass and writes final outputs to `cleaned_data/`, then produces load labels in `load_cleaned_data/`, and trains an XGBoost classifier.

## Files
- `clean_data.py` — unified one-shot cleaner (headers + filtering + broken sensor handling).
- `load_mask.py` — cleans load sensor signal and writes `*_forklift_load_cleaned.csv`.
- `train_XGBoost.py` — trains an XGBoost model on the load-cleaned data and saves artifacts.

## What `clean_data.py` does
1) Read every `*.csv` in `data/` (semicolon-delimited, no headers).
2) Add headers, drop consecutive redundant `OnDuty=0` rows.
3) Convert `Timestamp`/`Height` to numeric, drop NA, sort by timestamp.
4) Skip non-forklifts (no non-zero height readings).
5) If >10% of rows have `Height` > 7.0 m, mark as broken and write `<stem>_forklift_broken_height.csv` (no further cleaning).
6) Otherwise clean heights by removing rows above 7.0 m, cast `Timestamp` to int64, and write `<stem>_forklift.csv` to `cleaned_data/`.

## Config (constants in `clean_data.py`)
- `MIN_HEIGHT` / `MAX_HEIGHT`: bounds (defaults 0.0 / 7.0).
- `BROKEN_HEIGHT_THRESHOLD`: fraction above max that marks a broken sensor (default 0.10).
- `RAW_DIR` / `OUTPUT_DIR`: input/output folders (defaults `data/` and `cleaned_data/`).

## How to run
```
python clean_data.py
```

Ensure Python and pandas are installed; outputs will appear in `cleaned_data/`. Broken-sensor files are flagged in the filename and log.

## Load cleaning
Run `python load_mask.py` to produce `load_cleaned_data/` with `Load_Cleaned` and `LoadChange`, plus a summary CSV.

## Model training
Run `python train_XGBoost.py` to train a fixed-parameter XGBoost classifier (best params baked in) on `load_cleaned_data/`. Artifacts are written to `artifacts/` (`xgboost_load_model.json` and `load_prediction_results.png`).
