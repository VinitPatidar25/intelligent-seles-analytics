
import pandas as pd
import plotly.express as px



def get_chart_columns(data):
    """Identify usable numeric, categorical, and date columns."""
    result = {"numeric": [], "categorical": [], "date": []}
    date_hints = ("date", "day", "timestamp", "datetime")

    for column in data.columns:
        series = data[column]

        if pd.api.types.is_bool_dtype(series):
            result["categorical"].append(column)
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            result["date"].append(column)
            continue

        if pd.api.types.is_numeric_dtype(series):
            result["numeric"].append(column)
            continue

        has_date_hint = any(
            hint in str(column).lower() for hint in date_hints
        )

        if has_date_hint and series.notna().any():
            parsed = pd.to_datetime(series, errors="coerce")
            valid_count = parsed.notna().sum()
            non_missing_count = series.notna().sum()

            if valid_count / non_missing_count >= 0.8:
                result["date"].append(column)
                continue

        result["categorical"].append(column)

    return result



def create_chart(
    data,
    chart_type,
    x_column=None,
    y_column=None,
    aggregation="Sum",
):
    """Build a Plotly chart from the selected columns."""
    if data.empty:
        raise ValueError("No rows match the current filters.")

    if chart_type == "Histogram":
        if x_column not in data.columns:
            raise ValueError("Select a valid numeric column.")

        values = pd.to_numeric(data[x_column], errors="coerce")
        chart_data = pd.DataFrame({x_column: values}).dropna()

        if chart_data.empty:
            raise ValueError("The selected column has no numeric values.")

        return px.histogram(
            chart_data,
            x=x_column,
            title=f"Distribution of {x_column}",
            labels={x_column: x_column},
        )

    if chart_type == "Scatter":
        if x_column not in data.columns or y_column not in data.columns:
            raise ValueError("Select valid columns for both axes.")

        chart_data = data[[x_column, y_column]].copy()
        chart_data[x_column] = pd.to_numeric(
            chart_data[x_column], errors="coerce"
        )
        chart_data[y_column] = pd.to_numeric(
            chart_data[y_column], errors="coerce"
        )
        chart_data = chart_data.dropna()

        if chart_data.empty:
            raise ValueError("The selected columns have no valid numeric pairs.")

        return px.scatter(
            chart_data,
            x=x_column,
            y=y_column,
            title=f"{y_column} vs {x_column}",
        )

    if chart_type in ("Bar", "Line"):
        if x_column not in data.columns or y_column not in data.columns:
            raise ValueError("Select valid columns for both axes.")

        chart_data = data[[x_column, y_column]].copy()
        chart_data[y_column] = pd.to_numeric(
            chart_data[y_column], errors="coerce"
        )
        chart_data = chart_data.dropna(subset=[y_column])

        if chart_type == "Line":
            chart_data[x_column] = pd.to_datetime(
                chart_data[x_column], errors="coerce"
            )
            chart_data = chart_data.dropna(subset=[x_column])

        if chart_data.empty:
            raise ValueError("No valid data is available for this chart.")

        if aggregation == "Average":
            grouped = chart_data.groupby(
                x_column, as_index=False, dropna=False
            )[y_column].mean()
        else:
            grouped = chart_data.groupby(
                x_column, as_index=False, dropna=False
            )[y_column].sum()

        if chart_type == "Line":
            grouped = grouped.sort_values(x_column)
            return px.line(
                grouped,
                x=x_column,
                y=y_column,
                markers=True,
                title=f"{aggregation} of {y_column} over {x_column}",
            )

        grouped = grouped.sort_values(y_column, ascending=False)
        return px.bar(
            grouped,
            x=x_column,
            y=y_column,
            title=f"{aggregation} of {y_column} by {x_column}",
        )

    raise ValueError("Unsupported chart type.")
