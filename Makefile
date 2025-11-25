UV ?= uv
UV_RUN ?= $(UV) run
PYTHON ?= $(UV_RUN) python
DAGSTER_MODULE ?= pipeline.assets
MLFLOW_BACKEND ?= mlruns
MLFLOW_PORT ?= 5000

.PHONY: pipeline clean_data features train evaluate dagster mlflow dashboards help check-uv

# Run the full pipeline: cleaning -> feature engineering -> training -> evaluation
pipeline: evaluate

# Stage 1 & 2: raw cleaning and load masking
clean_data: check-uv
	$(PYTHON) -c "from pipeline.assets.cleaning import cleaning_pipeline; from pipeline.config import CLEANING_CONFIG, PREPROCESSING_CONFIG; cleaning_pipeline.run(CLEANING_CONFIG); cleaning_pipeline.run_load_cleaning(PREPROCESSING_CONFIG)"

# Stage 3: feature engineering
features: clean_data
	$(UV_RUN) -m pipeline.assets.engineered_data.engineered_data

# Stage 4: train XGBoost model (creates splits if missing)
train: features
	$(UV_RUN) -m pipeline.assets.XGBoost_training.train_XGBoost

# Stage 5: standalone evaluation of saved model + plots/metrics
evaluate: train
	$(UV_RUN) -m pipeline.assets.evaluating.evaluate_model

# Run Dagster UI (assets at $(DAGSTER_MODULE)); leave running in terminal
dagster: check-uv
	$(UV_RUN) dagster dev -m $(DAGSTER_MODULE)

# Launch MLflow UI pointing at local runs dir
mlflow: check-uv
	$(UV_RUN) mlflow ui --backend-store-uri $(MLFLOW_BACKEND) --port $(MLFLOW_PORT)

# Run Dagster UI and MLflow UI together (Ctrl+C to stop both)
dashboards: check-uv
	@echo "Starting Dagster (module: $(DAGSTER_MODULE)) and MLflow UI (backend: $(MLFLOW_BACKEND), port: $(MLFLOW_PORT))"
	@DAGSTER_CMD="$(UV_RUN) dagster dev -m $(DAGSTER_MODULE)"; \
	MLFLOW_CMD="$(UV_RUN) mlflow ui --backend-store-uri $(MLFLOW_BACKEND) --port $(MLFLOW_PORT)"; \
	$$DAGSTER_CMD & DAGSTER_PID=$$!; \
	$$MLFLOW_CMD & MLFLOW_PID=$$!; \
	trap 'kill $$DAGSTER_PID $$MLFLOW_PID' EXIT; \
	wait

check-uv:
	@command -v $(UV) >/dev/null || { echo "uv not found; install from https://docs.astral.sh/uv/"; exit 1; }

# Show available commands
help:
	@echo "Usage: make <target>"
	@echo "Uses uv to run all commands (override UV=... or UV_RUN=...)"
	@echo ""
	@echo "Pipeline:"
	@echo "  clean_data   Run cleaning and load masking (writes to cleaned_data/)"
	@echo "  features     Run feature engineering (depends on clean_data)"
	@echo "  train        Train XGBoost model and log to MLflow (depends on features)"
	@echo "  evaluate     Evaluate saved model and write plots/metrics (depends on train)"
	@echo "  pipeline     Full pipeline: clean_data -> features -> train -> evaluate"
	@echo ""
	@echo "Orchestration:"
	@echo "  dagster      Launch Dagster UI with assets module ($(DAGSTER_MODULE))"
	@echo "  mlflow       Launch MLflow UI (backend $(MLFLOW_BACKEND), port $(MLFLOW_PORT))"
	@echo "  dashboards   Launch Dagster and MLflow UIs together (Ctrl+C stops both)"
