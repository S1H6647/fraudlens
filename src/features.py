"""V1 features calculated from strictly earlier observed transactions."""
import numpy as np
import pandas as pd

RAW_COLUMNS = ["transaction_id", "user_id", "timestamp", "amount", "merchant_category",
               "country", "device_id", "channel", "hours_since_prev_txn"]


def build_features(transaction: dict, history: pd.DataFrame) -> dict:
    past = history.loc[
        history.user_id.eq(transaction["user_id"])
        & history.timestamp.lt(transaction["timestamp"])
        & history.transaction_id.ne(transaction["transaction_id"])
    ].sort_values(["timestamp", "transaction_id"])
    count = len(past)
    previous = past.iloc[-1] if count else None
    mean_amount = float(past.amount.mean()) if count else np.nan
    device_count = int(past.device_id.eq(transaction["device_id"]).sum())
    country_count = int(past.country.eq(transaction["country"]).sum())
    merchant_count = int(past.merchant_category.eq(transaction["merchant_category"]).sum())
    combo_count = int((past.device_id.eq(transaction["device_id"])
                       & past.country.eq(transaction["country"])).sum())
    row = {k: transaction[k] for k in [
        "timestamp", "amount", "merchant_category", "country", "channel", "hours_since_prev_txn"
    ]}
    row.update({
        "time_since_last_txn": transaction["timestamp"] - previous.timestamp if count else np.nan,
        "avg_prev_time_gap": past.timestamp.diff().tail(10).mean(),
        "user_transaction_count": count,
        "transactions_prev_10": min(count, 10),
        "previous_amount": previous.amount if count else np.nan,
        "user_avg_amount_prev": mean_amount,
        "user_max_amount_prev": past.amount.max() if count else np.nan,
        "amount_vs_user_avg": transaction["amount"] / (mean_amount + 1e-6),
        "amount_change": transaction["amount"] - previous.amount if count else np.nan,
        "rolling_avg_amount_10": past.amount.tail(10).mean(),
        "device_previous_count": device_count,
        "is_new_device": int(device_count == 0),
        "country_previous_count": country_count,
        "is_new_country": int(country_count == 0),
        "previous_country": previous.country if count else np.nan,
        "country_changed": int(count > 0 and transaction["country"] != previous.country),
        "merchant_previous_count": merchant_count,
        "is_new_merchant_category": int(merchant_count == 0),
        "device_country_previous_count": combo_count,
        "new_device_country_combo": int(combo_count == 0),
    })
    return row
