"""Small local API for the saved V1 fraud model."""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field
from src.features import RAW_COLUMNS, build_features
from src.runtime_assets import ensure_runtime_assets


MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts/xgb_recall_055.joblib"

DATA_PATH = Path(__file__).resolve().parents[1] / "datasets/train.csv"


class Transaction(BaseModel):
    """Nine raw fields; the server calculates historical features."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    transaction_id: int
    user_id: int
    timestamp: float = Field(ge=0)
    amount: float = Field(ge=0)
    merchant_category: int
    country: int
    device_id: int
    channel: int = Field(ge=0, le=1)
    hours_since_prev_txn: float = Field(ge=0)


class Prediction(BaseModel):
    transaction_id: int
    history_transactions: int
    fraud_score: float
    threshold: float
    flag_for_review: bool
    decision: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_path, model_path = ensure_runtime_assets(DATA_PATH, MODEL_PATH)
    saved = joblib.load(model_path)
    history = pd.read_csv(data_path, usecols=RAW_COLUMNS)
    if history.isna().any().any() or not history.transaction_id.is_unique:
        raise RuntimeError("History must have complete fields and unique transaction IDs.")
    probe = build_features(dict.fromkeys(RAW_COLUMNS, 0), history.iloc[:0])
    if set(saved["feature_columns"]) != set(probe):
        raise RuntimeError("Saved model feature columns do not match the API schema.")
    if not 0 <= float(saved["threshold"]) <= 1:
        raise RuntimeError("Saved model threshold must be between 0 and 1.")
    app.state.saved_model = saved
    app.state.history = history
    # Index once so requests inspect only the submitted user's history.
    app.state.history_indices = history.groupby("user_id", sort=False).indices
    try:
        yield
    finally:
        del app.state.saved_model
        del app.state.history
        del app.state.history_indices


app = FastAPI(
    title="FraudLens",
    description=(
        "Testing API for V1 XGBoost. Supply nine raw transaction fields; historical features are calculated automatically. "
        "Scores are experimental; a review flag is not proof of fraud."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "saved_model")}


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction, request: Request):
    saved = request.app.state.saved_model
    indices = request.app.state.history_indices.get(transaction.user_id, [])
    history = request.app.state.history.iloc[indices]
    row = build_features(transaction.model_dump(), history)
    frame = pd.DataFrame([row], columns=saved["feature_columns"])
    score = float(saved["pipeline"].predict_proba(frame)[0, 1])
    threshold = float(saved["threshold"])
    flagged = score >= threshold
    return Prediction(
        transaction_id=transaction.transaction_id,
        history_transactions=row["user_transaction_count"],
        fraud_score=score,
        threshold=threshold,
        flag_for_review=flagged,
        decision="REVIEW" if flagged else "NOT_FLAGGED",
    )
