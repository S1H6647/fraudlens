import pandas as pd

REQUIRED_COLUMNS = {
    "step",
    "transaction_type",
    "amount",
    "origin_account",
    "origin_balance_before",
    "origin_balance_after",
    "destination_account",
    "destination_balance_before",
    "destination_balance_after",
    "fraud",
    "flagged_fraud",
}

NON_NULL_COLUMNS = {
    "step",
    "transaction_type",
    "amount",
    "origin_account",
    "destination_account",
    "fraud",
}

BINARY_COLUMNS = {
    "fraud",
    "flagged_fraud",
}

BALANCE_COLUMNS = {
    "origin_balance_before",
    "origin_balance_after",
    "destination_balance_before",
    "destination_balance_after",
}

STRING_COLUMNS = {
    "transaction_type",
    "origin_account",
    "destination_account",
}


def validate_transactions(df: pd.DataFrame) -> None:
    errors: list[str] = []

    missing_columns = REQUIRED_COLUMNS - set(df)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        errors.append("Dataset contains no transactions")

    for column in sorted(NON_NULL_COLUMNS):
        null_count = int(df[column].isna().sum())

        if null_count > 0:
            errors.append(f"Column '{column}' contains {null_count} null value(s)")

    if not is_integer_dtype(df["step"]):
        errors.append("Column 'step' must have an integer dtype")
    elif (df["step"] < 1).any():
        errors.append("Column 'step' must contain values greater than or equal to 1")

    if not is_numeric_dtype(df["amount"]):
        errors.append("Column 'amount' must have a numeric dtype")
    elif (df["amount"] < 0).any():
        errors.append("Column 'amount' contains negative values")

    for column in sorted(BALANCE_COLUMNS):
        if not is_numeric_dtype(df[column]):
            errors.append(f"Column '{column}' must have a numeric dtype")
        elif (df[column] < 0).any():
            errors.append(f"Column '{column}' contains negative values")

    for column in sorted(BINARY_COLUMNS):
        if not is_integer_dtype(df[column]):
            errors.append(f"Column '{column}' must have an integer dtype")

        invalid_values = set(df[column].dropna().unique()) - {0, 1}  # pyright: ignore[reportAny]

        if invalid_values:
            formatted_values = ", ".join(
                sorted((repr(value) for value in invalid_values))
            )
            errors.append(
                f"Column '{column}' contains invalid values: {formatted_values}"
            )

    for column in sorted(STRING_COLUMNS):
        if not is_string_dtype(df[column]):
            errors.append(f"Column '{column}' must have a string dtype")

        empty_count = int(df[column].fillna("").astype(str).str.strip().eq("").sum())

        if empty_count > 0:
            errors.append(f"Column '{column}' contains {empty_count} empty value(s)")

    duplicate_count: int = int(df.duplicated().sum())  # pyright: ignore[reportAny]

    if duplicate_count > 0:
        errors.append(
            f"Dataset contains {duplicate_count} duplicate transaction row(s)"
        )

    if errors:
        details = "\n- ".join(errors)
        raise ValueError(f"Transaction validation failed:\n- {details}")


def is_integer_dtype(df: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(df)


def is_numeric_dtype(df: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(df)


def is_string_dtype(df: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(df)
