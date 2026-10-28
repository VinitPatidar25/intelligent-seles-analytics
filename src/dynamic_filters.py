
import pandas as pd


DATE_NAME_HINTS = ("date", "day", "timestamp", "datetime")


def detect_column_types(data):
    """
    Classify columns using their data types and values.
    Returns a dictionary with categorical, numeric, and date columns.
    """
    result = {
        "categorical": [],
        "numeric": [],
        "date": [],
    }

    for column in data.columns:
        series = data[column]

        # Avoid treating Boolean values as numeric measurements.
        if pd.api.types.is_bool_dtype(series):
            result["categorical"].append(column)
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            result["date"].append(column)
            continue

        if pd.api.types.is_numeric_dtype(series):
            result["numeric"].append(column)
            continue

        # Try date detection for text columns.
        # Column names provide a clue, but values must also parse as dates.
        name_hint = any(
            hint in str(column).lower()
            for hint in DATE_NAME_HINTS
        )

        if name_hint and series.notna().any():
            parsed = pd.to_datetime(series, errors="coerce")
            valid_ratio = parsed.notna().sum() / series.notna().sum()

            if valid_ratio >= 0.8:
                result["date"].append(column)
                continue

        result["categorical"].append(column)

    return result


def apply_filters(data, filters):
    """
    Apply filters to a copy of the original DataFrame.

    Supported filter types:
    - category: {"type": "category", "values": [...]}
    - numeric: {"type": "numeric", "min": number, "max": number}
    - date: {"type": "date", "start": date, "end": date}
    """
    filtered = data.copy()

    for column, settings in filters.items():
        if column not in filtered.columns:
            continue

        filter_type = settings.get("type")

        if filter_type == "category":
            selected = settings.get("values", [])

            # Empty selection means no restriction.
            if selected:
                filtered = filtered[
                    filtered[column].isin(selected)
                ]

        elif filter_type == "numeric":
            minimum = settings.get("min")
            maximum = settings.get("max")

            values = pd.to_numeric(
                filtered[column], errors="coerce"
            )

            mask = values.notna()

            if minimum is not None:
                mask &= values >= minimum

            if maximum is not None:
                mask &= values <= maximum

            filtered = filtered[mask]

        elif filter_type == "date":
            start = settings.get("start")
            end = settings.get("end")

            dates = pd.to_datetime(
                filtered[column], errors="coerce"
            )

            mask = dates.notna()

            if start is not None:
                mask &= dates.dt.date >= pd.Timestamp(start).date()

            if end is not None:
                mask &= dates.dt.date <= pd.Timestamp(end).date()

            filtered = filtered[mask]

    return filtered.copy()
