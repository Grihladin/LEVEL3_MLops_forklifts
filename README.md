# Forklift Telemetry Pipeline

Clean raw forklift telemetry, mask noisy load signals, and train/evaluate an XGBoost classifier for the loaded vs. unloaded state. All intermediate files stay in the repo, and Dagster assets can orchestrate the steps end-to-end.

## Repository layout
- `data/` — raw semicolon-delimited CSVs: `FFFID,Height,Load,OnDuty,Timestamp,Latitude,Longitude,Speed`.
- `cleaned_data/` — height-cleaned outputs and `_broken_height` files from stage 1, plus load-masked files and `cleaning_summary.csv` from stage 2.
- `engineered_data/` — feature-engineered outputs and `splits/` (train/test) written during training.
- `artifacts/` — trained model (`xgboost_load_model.json`), plots, metrics JSON, and local MLflow tracking under `artifacts/mlruns/`.
- `pipeline/assets/` — reusable steps: `cleaning/cleaning_pipeline.py`, `engineered_data/engineered_data.py`, `XGBoost_training/train_XGBoost.py`, `evaluating/evaluate_model.py`, plus `assets.py` for Dagster asset wiring.
- `pyproject.toml` / `uv.lock` — Python 3.13 project definition and locked dependencies.
- `pipeline/config/config.py` — single source for paths, thresholds, and model hyperparameters.

## Setup
- Python 3.13 (see `.python-version`).
- Install deps with `uv sync` (preferred) or `pip install -e .` to read `pyproject.toml`.
- Place raw CSVs in `data/` before running the pipeline.

## Configuration
- Edit `pipeline/config/config.py` to adjust paths, cleaning thresholds, masking rules (durations/height/speed), feature flags, and XGBoost hyperparameters.
- Dagster assets, the FastAPI server, and the training/evaluation scripts all read from this config so changes propagate without touching individual modules.

## Data flow (standalone)
1) **Height cleaning + load masking** — `pipeline/assets/cleaning/cleaning_pipeline.py`  
   Adds headers, removes repeated `OnDuty=0`, drops invalid `Timestamp/Height`, sorts by time. Skips non-forklifts (no non-zero height). Marks broken sensors if >10% readings exceed `MAX_HEIGHT` (7.0 m) and writes `<id>_forklift_broken_height.csv`; otherwise clamps to `[MIN_HEIGHT, MAX_HEIGHT]` and writes `<id>_forklift.csv` in `cleaned_data/`. Then masks load noise (height <0.05 m, remove events <30 s or >2 h) and writes `<id>_forklift_load_cleaned.csv` plus `cleaning_summary.csv` in `cleaned_data/`.  
   Run:
   ```bash
   uv run python -m pipeline.assets.cleaning.cleaning_pipeline
   ```

2) **Feature engineering** — `pipeline/assets/engineered_data/engineered_data.py`  
   Takes load-masked files in `cleaned_data/` (with `Load_Cleaned` already present), computes deltas/rates via `run_feature_engineering`, and writes `<id>_forklift_features.csv` to `engineered_data/`.  
   Run:
   ```bash
   uv run python -m pipeline.assets.engineered_data.engineered_data
   ```

3) **Train XGBoost** — `pipeline/assets/XGBoost_training/train_XGBoost.py`  
   Features: base signals plus deltas/rates (`DeltaHeight`, `DeltaSpeed`, `DeltaTimeSeconds`, `HeightChangeRate`, `TimeSinceLoadChange`). Creates stratified train/test splits (if missing) under `engineered_data/splits/`, uses fixed hyperparameters, and derives `scale_pos_weight` from the training split to balance classes automatically. Saves the model to `artifacts/xgboost_load_model.json`, plot `load_prediction_results.png`, and logs to MLflow in `artifacts/mlruns/`.  
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
  uv run dagster asset materialize cleaned_data engineered_data trained_model evaluated_model -m pipeline.assets
  ```
- Outputs stay in `cleaned_data/`, `engineered_data/`, and `artifacts/` with metadata attached in Dagster.

## FastAPI inference server
- Start the API (loads `artifacts/xgboost_load_model.json` once):
  ```bash
  uv run uvicorn api.main:app --port 4444
  ```
- Health check: `GET /health`
- Predict (batched; suitable for ~10–20 forklifts sending every ~2s):
  ```bash
  curl -X POST http://localhost:4444/predict \
    -H "Content-Type: application/json" \
    -d '{
          "readings": [
            {"forklift_id": "F1", "timestamp_ms": 1730000000000, "height": 1.8, "speed": 6.2, "on_duty": 1},
            {"forklift_id": "F2", "timestamp_ms": 1730000000500, "height": 0.3, "speed": 12.5, "on_duty": 1}
          ]
        }'
  ```
- Response fields: per reading you get `loaded_probability` (0–1) and `loaded` (boolean), plus echoed `forklift_id`/`timestamp_ms`.

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
- Model: XGBoost binary classifier (max_depth=8, learning_rate=0.2, n_estimators=150, min_child_weight=5, subsample=0.8, colsample_bytree=0.8) with `scale_pos_weight` computed from the observed class imbalance during training.

## Current results
- `artifacts/evaluation_metrics.json` reports ~0.81 accuracy on the held-out split (class imbalance handled via dynamic `scale_pos_weight`).
- Plots: `artifacts/load_prediction_results.png`, `artifacts/confusion_matrix.png`, `artifacts/probability_distribution.png` for quick inspection.

## Adding new data
Drop new semicolon-delimited raw CSVs (no header, same column order) into `data/`, then rerun steps 1–4 or materialize via Dagster to refresh cleaned outputs, splits, and model artifacts.
