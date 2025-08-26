# app.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import json
import os
import uvicorn
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = 'outputs/crop_yield_pipeline_latest.joblib'
RESID_STATS = 'outputs/residual_stats.json'
META = 'outputs/model_metadata.json'


app = FastAPI(title="Crop Yield Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_model():
    global pipeline, resid_stats, metadata, estimator, preproc, estimator_step_name
    pipeline = joblib.load(MODEL_PATH)
    resid_stats = {}
    metadata = {}
    estimator = None
    preproc = None
    estimator_step_name = None
    if Path(RESID_STATS).exists():
        with open(RESID_STATS, 'r') as f:
            resid_stats = json.load(f)
    if Path(META).exists():
        with open(META, 'r') as f:
            metadata = json.load(f)
        estimator_step_name = metadata.get('estimator_step_name')
    if estimator_step_name is None:
        for candidate in ('estimator', 'model'):
            if candidate in pipeline.named_steps:
                estimator_step_name = candidate
                break
        if estimator_step_name is None:
            estimator_step_name = list(pipeline.named_steps.keys())[-1]
    estimator = pipeline.named_steps.get(estimator_step_name, None)
    preproc = pipeline.named_steps.get('preprocessor', None)

class PredictRequest(BaseModel):
    Crop: str
    Crop_Year: Optional[int] = None
    Season: Optional[str] = None
    State: Optional[str] = None
    Area: float
    Annual_Rainfall: Optional[float] = None
    Fertilizer: Optional[float] = None
    Pesticide: Optional[float] = None

class PredictResponse(BaseModel):
    prediction: float
    lower_95: float
    upper_95: float
    model: Optional[str] = None

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    row = pd.DataFrame([req.dict()])
    pred = float(pipeline.predict(row)[0])

    if resid_stats and 'resid_std' in resid_stats:
        resid_std = resid_stats['resid_std']
        lower = pred - 1.96 * resid_std
        upper = pred + 1.96 * resid_std
    else:
        lower, upper = pred, pred
    try:
        if estimator is not None and hasattr(estimator, 'estimators_'):
            if preproc is not None:
                X_trans = preproc.transform(row)
                tree_preds = np.array([t.predict(X_trans) for t in estimator.estimators_])
                mean_pred = float(np.mean(tree_preds))
                lower = float(np.percentile(tree_preds, 2.5))
                upper = float(np.percentile(tree_preds, 97.5))
                pred = mean_pred
    except Exception:
        pass

    return PredictResponse(prediction=float(pred), lower_95=float(lower), upper_95=float(upper), model=metadata.get('model_file'))

@app.get("/health")
def health():
    return {"status": "ok", "model": metadata.get('model_file')}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # Run the FastAPI app with uvicorn, binding to 0.0.0.0 to accept all incoming connections
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
