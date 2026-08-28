# Responsibilities:

# load raw dataset
# parse timestamps
# normalize obvious data types
# return DataFrame

from pathlib import Path

import pandas as pd

COLUMN_MAPPING = {
    "type": "transaction_type",
    "isFraud": "fraud",
    "isFlaggedFraud": "flagged_fraud",
    "nameOrig": "origin_account",
    "nameDest": "destination_account",
    "oldbalanceOrg": "origin_balance_before",
    "newbalanceOrig": "origin_balance_after",
    "oldbalanceDest": "destination_balance_before",
    "newbalanceDest": "destination_balance_after",
}


def load_transactions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Could not load transactions from {path}")

    df = pd.read_csv(path)

    df = df.rename(columns=COLUMN_MAPPING)

    df["fraud"] = df["fraud"].astype("Int8")
    df["flagged_fraud"] = df["flagged_fraud"].astype("Int8")

    return df
