# FraudLens

A notebook-based machine learning project exploring fraud detection from transaction details and customer behavior. It compares Random Forest and XGBoost, engineers historical features, and studies the tradeoff between catching fraud and flagging legitimate transactions.

The retained model is the tuned **XGBoost pipeline from [V1.ipynb](notebooks/V1.ipynb)**, with a validation-selected threshold targeting at least 55% recall. This is an experimental project: its current performance does not support automatic transaction blocking.

## Dataset

Data comes from Kaggle's [Behavioral Fraud Detection competition](https://www.kaggle.com/competitions/behavioral-fraud-detection/data). The competition describes synthetic transactions with probabilistic fraud labels based on deviations from user behavior.

Place the downloaded files in `datasets/`:

- `train.csv`: 182,125 labeled transactions, including 2,990 fraud cases (approximately 1.64%).
- `test.csv`: unlabeled competition transactions; these cannot provide local evaluation metrics.
- `sample_submission.csv`: competition submission template.

Fields include transaction and user identifiers, timestamp, amount, merchant category, country, device, channel, and a supplied previous-transaction gap. The training target is `label`.

## Repository layout

```text
notebooks/V1.ipynb             Feature engineering, training, tuning, evaluation, export
src/evaluate_model.py         Classification and ranking metrics
src/threshold.py              Validation-based F1 threshold selection
src/api.py                    FastAPI health and prediction endpoints
src/features.py               Raw transaction to V1 historical features
examples/transaction.json     Example prediction request
tests/test_api.py             API validation and artifact integration tests
datasets/                     Local competition CSVs
artifacts/xgb_recall_055.joblib Local fitted pipeline and threshold
pyproject.toml                Dependencies and Python requirement
uv.lock                       Dependency lockfile
```

Model artifacts are ignored by Git and must be generated locally. A small local API is included for testing the saved model; there is no deployed review service or production feature store.

## Setup and execution

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

From the repository root:

```bash
uv sync --locked --no-install-project
uv run --no-project .venv/bin/jupyter lab notebooks/V1.ipynb
```

`--no-install-project` installs dependencies without building the repository as a package. The current source layout is used directly by the notebook.

Before running the notebook:

1. Replace its machine-specific CSV path with your local path. With the kernel working directory set to `notebooks/`, use `pd.read_csv("../datasets/train.csv")`.
2. Keep the kernel working directory at `notebooks/` so the existing parent-directory import setup locates `src/`.
3. To export into the repository-level artifact directory, change the export cell to `artifact_dir = Path("../artifacts")`. The current `Path("artifacts")` is relative to the kernel directory and may save under `notebooks/artifacts/`.
4. Run cells in order. Randomized search performs 20 configurations across 3 folds, followed by refitting the winner.

## Run the local backend

From the repository root, with the saved model at `artifacts/xgb_recall_055.joblib`:

```bash
uv run --no-project .venv/bin/uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Open [interactive API docs](http://127.0.0.1:8000/docs) to test requests. `GET /health` checks model availability; `POST /predict` scores one transaction and returns `transaction_id`, `history_transactions`, `fraud_score`, the saved `threshold`, `flag_for_review`, and a `REVIEW` or `NOT_FLAGGED` decision.

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/transaction.json
```

Supply only these nine raw fields:

```json
{
  "transaction_id": 300000,
  "user_id": 3399,
  "timestamp": 2900000,
  "amount": 50.0,
  "merchant_category": 3,
  "country": 10,
  "device_id": 22279,
  "channel": 0,
  "hours_since_prev_txn": 1000
}
```

The example is illustrative, not a known fraud case. The backend loads `datasets/train.csv` once at startup, ignores its labels, and calculates all 26 V1 inputs using only that user's earlier transactions. Future rows, rows at the submitted timestamp, and rows with the submitted transaction ID are excluded. Earlier history is ordered by timestamp then transaction ID for deterministic ties. Unlike V1's arbitrary within-timestamp ordering, simultaneous transactions are not treated as prior observations of the submitted transaction.

For Render, set `FRAUD_HISTORY_URL` to your direct Hugging Face file URL, such as `https://huggingface.co/datasets/S1H6647/fraudlens/resolve/main/train.csv?download=true`. Set `FRAUD_MODEL_URL` to a direct `xgb_recall_055.joblib` URL when the model is not included in the image. For private repositories, set `HUGGINGFACE_TOKEN` as a secret environment variable. If these URLs are unset, local files are used.

Users without history receive zero counts, new-device/location flags, and missing historical amounts/gaps that the saved pipeline imputes. The supplied `hours_since_prev_txn` is passed through unchanged. Missing raw fields, extra fields, and invalid values are rejected. Requests do not append to history, alter CSVs, or retrain the model; restart the backend after updating the dataset. Scoring an existing training transaction is a demonstration, not independent model evaluation.

Run the API checks with:

```bash
uv run --no-project .venv/bin/python -m pytest tests/test_api.py -q
```

These integration checks use the local saved artifact and skip if it is absent.

## Modeling workflow

1. Inspect class balance and missing values.
2. Sort transactions by user and time, then construct previous-transaction, spending, device, country, and merchant-history features.
3. Exclude labels and raw transaction/user/device identifiers from model inputs.
4. Create stratified random training/validation/test splits of approximately 70%/15%/15%.
5. Fit imputation and categorical one-hot encoding within model pipelines.
6. Compare Random Forest and XGBoost; tune XGBoost with `RandomizedSearchCV`, scoring average precision.
7. Select an XGBoost-specific threshold on validation scores. The final export uses the highest-precision threshold meeting the 55% validation recall target.
8. Save the fitted pipeline, threshold, target recall, and expected feature columns together.

Earlier exploratory cells apply a Random Forest threshold to XGBoost. Those outputs are not the final model evaluation; use the later `best_xgb` recall-threshold section for the retained model.

## Retained model results

The saved XGBoost artifact uses 26 input features and threshold **0.10919876396656036**. The following are **validation results**, not an independent final test evaluation. The threshold was selected on these same validation observations.

| Metric | Validation result |
|---|---:|
| ROC-AUC | 0.5458 |
| Average precision (reported as PR-AUC) | 0.02285 |
| Accuracy | 51.69% |
| Precision | 1.88% |
| Recall | 55.46% |
| F1 | 0.03636 |

| Actual class | Predicted legitimate | Flagged for review |
|---|---:|---:|
| Legitimate | 13,873 | 12,997 |
| Fraud | 200 | 249 |

The model catches 249 of 449 fraud cases but flags approximately 48.37% of legitimate transactions. Its average precision is only modestly above the validation fraud prevalence of 1.64%. Lowering the threshold increases recall at a substantial false-alarm cost; it does not improve ranking quality.

## Load and use the saved model

From the repository root, after generating the artifact:

```python
import joblib

saved = joblib.load("artifacts/xgb_recall_055.joblib")

# engineered_transactions must contain the same historical features as V1.
X_new = engineered_transactions.loc[:, saved["feature_columns"]]
scores = saved["pipeline"].predict_proba(X_new)[:, 1]

results = engineered_transactions.copy()
results["fraud_score"] = scores
results["flag_for_review"] = scores >= saved["threshold"]
```

The artifact contains `pipeline`, `threshold`, `target_recall`, and `feature_columns`. It includes fitted imputation, encoding, and XGBoost, but **does not include historical feature construction**. The API performs that step using `src/features.py`; direct artifact callers must also construct those features. Weighted model scores are not established as calibrated fraud probabilities, and the recall target is not guaranteed on new data. Only load trusted joblib files.

## Limitations and conclusions

- Both the baseline and richer models show weak validation separation. The richer XGBoost model improved ranking metrics, but substantial overfitting remains: training ROC-AUC was approximately 0.797 versus 0.546 on validation.
- Random stratified splits do not establish performance on future transactions. A separate chronological audit evaluated a different Random Forest model; its results must not be attributed to this saved V1 XGBoost artifact.
- History reconstructed from a subset of transactions may be incomplete. An unseen device or country in the available data is not necessarily new in the full simulation.
- Timestamp units require confirmation; do not assume the supplied `hours_since_prev_txn` field is measured in hours solely from its name.
- Historical features require strict ordering, careful treatment of equal timestamps, and access only to previously observable events.
- API tests cover input validation, history exclusion rules, unseen users, rolling calculations, and agreement with direct saved-pipeline inference. They do not establish model quality or production readiness.

This project demonstrates an end-to-end exploratory modeling workflow and the practical limits of threshold tuning on a weak classifier. Further work would require stronger verified behavioral signals, chronological evaluation, and a review-capacity or false-positive budget before considering operational use.
