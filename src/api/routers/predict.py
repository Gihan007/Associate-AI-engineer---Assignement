import os
import time
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.joblib")

router = APIRouter()


class PredictRequest(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


class PredictResponse(BaseModel):
    prediction: int
    probability: float


model = None
model_loaded_at: float = 0.0


def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def get_model():
    global model, model_loaded_at
    model_mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0.0
    if model is None or model_mtime > model_loaded_at:
        model = load_model()
        model_loaded_at = time.time()
    return model


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    m = get_model()
    if m is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([payload.dict()])

    pred = m.predict(df)[0]
    proba = m.predict_proba(df)[0][1] if hasattr(m, "predict_proba") else 0.0
    return PredictResponse(prediction=int(pred), probability=float(proba))
