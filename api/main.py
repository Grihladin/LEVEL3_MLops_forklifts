"""FastAPI inference service for the forklift load classifier.

- Loads the trained XGBoost model from ``artifacts/xgboost_load_model.json``.
- Exposes a `/predict` endpoint that accepts batched telemetry and returns
  loaded/unloaded predictions and probabilities.
- Designed for ~10-20 forklifts sending readings every ~2 seconds.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "artifacts" / "xgboost_load_model.json"
FEATURE_COLUMNS = ["Height", "Speed", "OnDuty", "Height_Speed_Interaction", "Is_Moving"]

app = FastAPI(
    title="Forklift Load Inference",
    description="Predicts loaded vs. unloaded state from forklift telemetry.",
    version="0.1.0",
)

_model: Optional[xgb.XGBClassifier] = None


class Telemetry(BaseModel):
    forklift_id: Optional[str] = Field(None, description="Identifier for the forklift.")
    timestamp_ms: Optional[int] = Field(None, description="Epoch timestamp in milliseconds.")
    height: float = Field(..., description="Mast height in meters.")
    speed: float = Field(..., description="Speed in km/h.")
    on_duty: int = Field(..., ge=0, le=1, description="OnDuty flag (0 or 1).")


class Prediction(BaseModel):
    forklift_id: Optional[str]
    timestamp_ms: Optional[int]
    loaded_probability: float
    loaded: bool


class PredictRequest(BaseModel):
    readings: List[Telemetry]


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    model_path: str


def _load_model() -> xgb.XGBClassifier:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Train the model before starting the API."
            )
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        _model = model
    return _model


def _build_features(readings: list[Telemetry]) -> pd.DataFrame:
    base = pd.DataFrame(
        [
            {
                "Height": item.height,
                "Speed": item.speed,
                "OnDuty": int(item.on_duty),
            }
            for item in readings
        ]
    )
    base["Height_Speed_Interaction"] = base["Height"] * base["Speed"]
    base["Is_Moving"] = (base["Speed"] > 1.0).astype(int)
    return base[FEATURE_COLUMNS].fillna(0)


@app.get("/health")
def health() -> dict:
    """Simple readiness probe."""
    model_present = MODEL_PATH.exists()
    return {"status": "ok", "model_path": str(MODEL_PATH), "model_present": model_present}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not payload.readings:
        raise HTTPException(status_code=400, detail="No readings provided.")

    model = _load_model()
    features = _build_features(payload.readings)
    proba = model.predict_proba(features)[:, 1]
    classes = proba >= 0.5

    predictions = []
    for item, p, c in zip(payload.readings, proba, classes, strict=False):
        predictions.append(
            Prediction(
                forklift_id=item.forklift_id,
                timestamp_ms=item.timestamp_ms,
                loaded_probability=float(p),
                loaded=bool(c),
            )
        )

    return PredictResponse(predictions=predictions, model_path=str(MODEL_PATH))
