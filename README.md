# Forklift Telemetry Pipeline

Clean raw forklift telemetry, mask noisy load signals, and train/evaluate an XGBoost classifier for the loaded vs. unloaded state. All intermediate files stay in the repo, and Dagster assets can orchestrate the steps end-to-end.

## Repository layout
- `data/` — raw semicolon-delimited CSVs: `FFFID,Height,Load,OnDuty,Timestamp,Latitude,Longitude,Speed`.
- `cleaned_data/` — height-cleaned outputs and `_broken_height` files from stage 1.
- `load_cleaned_data/` — load-masked outputs, `cleaning_summary.csv`, and `splits/` (train/test) from stage 2.
- `artifacts/` — trained model (`xgboost_load_model.json`), plots, metrics JSON, and local MLflow tracking under `artifacts/mlruns/`.
- `pipeline/assets/` — reusable steps: `cleaning/cleaning_pipeline.py`, `preprocessing/preprocessing.py`, `XGBoost_training/train_XGBoost.py`, `evaluating/evaluate_model.py`, plus `assets.py` for Dagster asset wiring.
- `pyproject.toml` / `uv.lock` — Python 3.13 project definition and locked dependencies.

## Setup
- Python 3.13 (see `.python-version`).
- Install deps with `uv sync` (preferred) or `pip install -e .` to read `pyproject.toml`.
- Place raw CSVs in `data/` before running the pipeline.

## Data flow (standalone)
1) **Height cleaning** — `pipeline/assets/cleaning/cleaning_pipeline.py`  
   Adds headers, removes repeated `OnDuty=0`, drops invalid `Timestamp/Height`, sorts by time. Skips non-forklifts (no non-zero height). Marks broken sensors if >10% readings exceed `MAX_HEIGHT` (7.0 m) and writes `<id>_forklift_broken_height.csv`; otherwise clamps to `[MIN_HEIGHT, MAX_HEIGHT]` and writes `<id>_forklift.csv` in `cleaned_data/`.  
   Run:
   ```bash
   uv run python -m pipeline.assets.cleaning.cleaning_pipeline
   ```

2) **Load masking & splits** — `pipeline/assets/preprocessing/preprocessing.py`  
   Uses `<id>_forklift.csv`, ignores `_broken_height` files. Sets `Load` to zero when height < 0.05 m or speed > 15 km/h, removes events <30 s or >2 h, and writes `<id>_forklift_load_cleaned.csv`. Creates `cleaning_summary.csv` plus stratified `splits/train.csv` and `splits/test.csv`.  
   Run:
   ```bash
   uv run python -m pipeline.assets.preprocessing.preprocessing
   ```

3) **Train XGBoost** — `pipeline/assets/XGBoost_training/train_XGBoost.py`  
   Features: `Height`, `Speed`, `OnDuty`, `Height*Speed`, `Is_Moving`. Uses fixed hyperparameters with `scale_pos_weight` to balance classes. Saves the model to `artifacts/xgboost_load_model.json`, plot `load_prediction_results.png`, and logs to MLflow in `artifacts/mlruns/`.  
   Run:
   ```bash
   uv run python -m pipeline.assets.XGBoost_training.train_XGBoost
   ```

4) **Evaluate held-out split** — `pipeline/assets/evaluating/evaluate_model.py`  
   Loads `splits/test.csv` and the saved model, then writes `evaluation_metrics.json`, `confusion_matrix.png`, and `probability_distribution.png` in `artifacts/`.  
   Run:
   ```bash
   uv run python -m pipeline.assets.evaluating.evaluate_model
   ```

## Dagster orchestration
- Asset definitions live in `pipeline/assets/assets.py`.
- Start Dagster locally:
  ```bash
  uv run dagster dev -m pipeline.assets
  ```
- Materialize everything in order:
  ```bash
  uv run dagster asset materialize cleaned_data load_cleaned_data trained_model evaluated_model -m pipeline.assets
  ```
- Outputs stay in `cleaned_data/`, `load_cleaned_data/`, and `artifacts/` with metadata attached in Dagster.

## MLflow tracking
- Training and evaluation log runs to the local store at `artifacts/mlruns/` (set via a file URI).
- Launch the MLflow UI to browse params, metrics, and artifacts:
  ```bash
  uv run mlflow ui --backend-store-uri artifacts/mlruns --default-artifact-root artifacts
  ```
- Keep the UI command running in a terminal, then run training/eval to see new runs appear.

## Key parameters
- Stage 1: `MIN_HEIGHT=0.0`, `MAX_HEIGHT=7.0`, `BROKEN_HEIGHT_THRESHOLD=0.10`.
- Stage 2: `MIN_HEIGHT_THRESHOLD=0.05`, `MIN_SPEED_FOR_FILTERING=15`, `MIN_LOAD_DURATION=30`, `MAX_LOAD_DURATION=7200`.
- Model: XGBoost binary classifier (max_depth=8, learning_rate=0.2, n_estimators=150, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5.0).

## Current results
- `artifacts/evaluation_metrics.json` reports ~0.81 accuracy on the held-out split (class imbalance handled via `scale_pos_weight`).
- Plots: `artifacts/load_prediction_results.png`, `artifacts/confusion_matrix.png`, `artifacts/probability_distribution.png` for quick inspection.

## Adding new data
Drop new semicolon-delimited raw CSVs (no header, same column order) into `data/`, then rerun steps 1–4 or materialize via Dagster to refresh cleaned outputs, splits, and model artifacts.
