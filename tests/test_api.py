import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import DATA_PATH, MODEL_PATH, app
from src.features import RAW_COLUMNS, build_features


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def payload():
    return json.loads((ROOT / "examples/transaction.json").read_text())


@pytest.fixture
def client():
    if not MODEL_PATH.exists():
        pytest.skip("Export the local V1 model artifact before running API integration tests")
    with TestClient(app) as test_client:
        yield test_client


def test_prediction_matches_saved_pipeline(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    saved = joblib.load(MODEL_PATH)
    row = build_features(payload, pd.read_csv(DATA_PATH, usecols=RAW_COLUMNS))
    expected = saved["pipeline"].predict_proba(
        pd.DataFrame([row], columns=saved["feature_columns"])
    )[0, 1]
    result = response.json()
    assert result["fraud_score"] == pytest.approx(float(expected))
    assert result["threshold"] == saved["threshold"]
    assert result["flag_for_review"] == bool(expected >= saved["threshold"])
    assert result["decision"] == ("REVIEW" if result["flag_for_review"] else "NOT_FLAGGED")
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


@pytest.mark.parametrize("change", ["missing", "extra", "negative", "nonfinite"])
def test_invalid_features_rejected(client, payload, change):
    if change == "missing":
        del payload["amount"]
    elif change == "extra":
        payload["label"] = 1
    elif change == "negative":
        payload["amount"] = -1
    else:
        payload["amount"] = "NaN"
    assert client.post("/predict", json=payload).status_code == 422


def test_history_excludes_other_users_future_ties_and_same_id(payload):
    payload.update(transaction_id=99, user_id=1, timestamp=100, amount=60,
                   device_id=7, country=2, merchant_category=3)
    def row(**values):
        return {**payload, **values}
    history = pd.DataFrame([
        row(transaction_id=1, timestamp=10, amount=20),
        row(transaction_id=2, timestamp=30, amount=40, country=5),
        row(transaction_id=3, timestamp=200, amount=999),
        row(transaction_id=4, timestamp=100, amount=999),
        row(transaction_id=5, timestamp=20, user_id=2, amount=999),
        row(transaction_id=99, timestamp=5, amount=999),
    ])
    result = build_features(payload, history)
    assert result["user_transaction_count"] == 2
    assert result["time_since_last_txn"] == 70
    assert result["avg_prev_time_gap"] == 20
    assert result["previous_amount"] == 40
    assert result["user_avg_amount_prev"] == 30
    assert result["rolling_avg_amount_10"] == 30
    assert result["amount_change"] == 20
    assert result["amount_vs_user_avg"] == pytest.approx(60 / (30 + 1e-6))
    assert result["country_changed"] == 1
    assert result["device_previous_count"] == 2
    assert result["country_previous_count"] == 1
    assert result["device_country_previous_count"] == 1


def test_new_user_and_last_ten_history(payload):
    empty = pd.DataFrame(columns=RAW_COLUMNS)
    result = build_features(payload, empty)
    assert result["user_transaction_count"] == 0
    assert result["is_new_device"] == 1
    assert np.isnan(result["previous_amount"])
    assert np.isnan(result["avg_prev_time_gap"])
    history = pd.DataFrame([
        {**payload, "transaction_id": i, "timestamp": i * i, "amount": i}
        for i in range(1, 13)
    ])
    result = build_features(payload, history)
    assert result["transactions_prev_10"] == 10
    assert result["rolling_avg_amount_10"] == 7.5
    assert result["avg_prev_time_gap"] == 14


def test_unknown_user_prediction(client, payload):
    payload["user_id"] = -999
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["history_transactions"] == 0
