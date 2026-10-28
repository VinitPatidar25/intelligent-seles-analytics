
import pandas as pd


def validate_dataframe(data):
    """
    Check whether a DataFrame is usable for analysis.

    Returns:
        tuple[list[str], list[str]]:
            errors and warnings
    """
    errors = []
    warnings = []

    if not isinstance(data, pd.DataFrame):
        return ["The uploaded data is not a valid table."], warnings

    if data.empty:
        errors.append("The dataset contains no data rows.")
        return errors, warnings

    if len(data.columns) == 0:
        errors.append("The dataset contains no columns.")
        return errors, warnings

    # Detect blank or missing column names.
    blank_columns = [
        str(column)
        for column in data.columns
        if pd.isna(column) or str(column).strip() == ""
    ]

    if blank_columns:
        warnings.append(
            "Some columns have blank names. Rename them before analysis."
        )

    # Detect duplicate column names.
    if data.columns.duplicated().any():
        errors.append(
            "Duplicate column names were found. Rename them before analysis."
        )

    # Count missing cells.
    missing_cells = int(data.isna().sum().sum())

    if missing_cells:
        warnings.append(
            f"The dataset contains {missing_cells} missing cell(s)."
        )

    # Count completely duplicated rows.
    duplicate_rows = int(data.duplicated().sum())

    if duplicate_rows:
        warnings.append(
            f"The dataset contains {duplicate_rows} duplicate row(s)."
        )

    return errors, warnings


def get_quality_summary(data):
    """Return basic data-quality statistics."""
    return {
        "rows": len(data),
        "columns": len(data.columns),
        "missing_cells": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
    }
