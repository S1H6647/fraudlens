from pathlib import Path

import pandas as pd
import pytest

from fraudlens.data.loading import load_transactions


def test_load_transactions(tmp_path: Path) -> None:
    csv_path = tmp_path / "transactions.csv"
    csv_path.write_text(
        "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
        "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
        "1,PAYMENT,9839.64,C123,170136.0,160296.36,"
        "M456,0.0,0.0,0,0\n",
        encoding="utf-8",
    )

    result = load_transactions(csv_path)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

    assert "fraud" in result.columns
    assert "flagged_fraud" in result.columns
    assert "origin_account" in result.columns
    assert "destination_account" in result.columns

    assert "isFraud" not in result.columns
    assert "nameOrig" not in result.columns

    assert result["fraud"].dtype == "Int8"
    assert result["flagged_fraud"].dtype == "Int8"
    assert pd.api.types.is_integer_dtype(result["step"])


def test_load_transactions_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Could not load transactions"):
        load_transactions(missing_path)
