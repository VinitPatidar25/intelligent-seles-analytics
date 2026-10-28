import numpy as np
import pandas as pd


def prepare_time_series(data, date_column, value_column):
    """Clean and aggregate data into a date-sorted time series."""

    if data.empty:
        raise ValueError("No rows are available for forecasting.")

    if date_column not in data.columns:
        raise ValueError("The selected date column does not exist.")

    if value_column not in data.columns:
        raise ValueError("The selected numeric column does not exist.")

    if date_column == value_column:
        raise ValueError("Choose different date and numeric columns.")

    prepared = data[[date_column, value_column]].copy()

    prepared[date_column] = pd.to_datetime(
        prepared[date_column],
        errors="coerce",
    )

    prepared[value_column] = pd.to_numeric(
        prepared[value_column],
        errors="coerce",
    )

    prepared = prepared.dropna(
        subset=[date_column, value_column]
    )

    if prepared.empty:
        raise ValueError(
            "No valid date and numeric values were found."
        )

    prepared = prepared.groupby(
        date_column,
        as_index=False,
    )[value_column].sum()

    prepared = prepared.sort_values(date_column)

    prepared = prepared.rename(
        columns={
            date_column: "Date",
            value_column: "Actual",
        }
    )

    prepared = prepared.reset_index(drop=True)

    if len(prepared) < 2:
        raise ValueError(
            "At least 2 distinct dates with valid numeric values "
            "are required to prepare a time series."
        )
    return prepared
    
def forecast_linear_trend(time_series, periods=7):
    """Forecast future values using a linear trend.

    Expects a DataFrame with Date and Actual columns.
    Forecast periods represent future daily steps.
    """
    if time_series.empty:
        raise ValueError("No historical data is available to forecast.")

    if not {"Date", "Actual"}.issubset(time_series.columns):
        raise ValueError(
            "Input must contain Date and Actual columns."
        )

    if not isinstance(periods, int) or isinstance(periods, bool) or periods < 1:
        raise ValueError("Forecast periods must be a positive integer.")

    history = time_series[["Date", "Actual"]].copy()
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    history["Actual"] = pd.to_numeric(history["Actual"], errors="coerce")
    history = history.dropna(subset=["Date", "Actual"])
    history = history.sort_values("Date")

    if len(history) < 2:
        raise ValueError(
            "At least 2 valid historical observations are required."
        )

    if history["Date"].duplicated().any():
        raise ValueError(
            "Historical data must have one observation per date. "
            "Run prepare_time_series first."
        )

    # Use equally spaced time steps for fitting the trend.
    x = np.arange(len(history), dtype=float)
    y = history["Actual"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(
        len(history),
        len(history) + periods,
        dtype=float,
    )

    predictions = intercept + slope * future_x

    last_date = history["Date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=periods,
        freq="D",
    )

    forecast = pd.DataFrame({
        "Date": future_dates,
        "Forecast": predictions,
    })

    return forecast

def evaluate_linear_trend(time_series, test_size=1):
    """Evaluate a linear-trend forecast on held-out observations."""
    import numpy as np
    import pandas as pd

    required_columns = {"Date", "Actual"}

    if not required_columns.issubset(time_series.columns):
        raise ValueError("Time series must contain Date and Actual columns.")

    if not isinstance(test_size, int) or isinstance(test_size, bool):
        raise ValueError("Test size must be a positive integer.")

    if test_size < 1:
        raise ValueError("Test size must be at least 1.")

    if time_series["Date"].duplicated().any():
        raise ValueError("Time series must not contain duplicate dates.")

    if len(time_series) < 3:
        raise ValueError(
            "At least 3 observations are required for evaluation."
        )

    if test_size >= len(time_series) - 1:
        raise ValueError(
            "Keep at least 2 observations for training and 1 for testing."
        )

    data = time_series.sort_values("Date").reset_index(drop=True).copy()
    data["Actual"] = pd.to_numeric(data["Actual"], errors="coerce")

    if data["Actual"].isna().any():
        raise ValueError("Actual values must all be valid numeric values.")

    if not np.isfinite(data["Actual"].to_numpy(dtype=float)).all():
        raise ValueError("Actual values must be finite numbers.")

    train_data = data.iloc[:-test_size].copy()
    test_data = data.iloc[-test_size:].copy()

    predictions = forecast_linear_trend(
        train_data,
        periods=test_size,
    )

    actual_values = test_data["Actual"].to_numpy(dtype=float)
    predicted_values = predictions["Forecast"].to_numpy(dtype=float)

    errors = actual_values - predicted_values

    results = pd.DataFrame({
        "Date": test_data["Date"].to_numpy(),
        "Actual": actual_values,
        "Predicted": predicted_values,
        "Error": errors,
        "Absolute Error": np.abs(errors),
    })

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "results": results,
    }

def evaluate_last_value_baseline(time_series, test_size=1):
    """Evaluate a baseline that repeats the last training value."""
    required_columns = {"Date", "Actual"}

    if not required_columns.issubset(time_series.columns):
        raise ValueError("Time series must contain Date and Actual columns.")

    if not isinstance(test_size, int) or isinstance(test_size, bool) or test_size < 1:
        raise ValueError("Test size must be a positive integer.")
    if time_series["Date"].duplicated().any():
        raise ValueError("Time series must not contain duplicate dates.")

    if len(time_series) < 3:
        raise ValueError(
            "At least 3 observations are required for evaluation."
        )

    if test_size >= len(time_series) - 1:
        raise ValueError(
            "Keep at least 2 observations for training and 1 for testing."
        )

    data = time_series.sort_values("Date").reset_index(drop=True).copy()
    data["Actual"] = pd.to_numeric(data["Actual"], errors="coerce")

    if data["Actual"].isna().any():
        raise ValueError("Actual values must all be valid numeric values.")

    if not np.isfinite(data["Actual"].to_numpy(dtype=float)).all():
        raise ValueError("Actual values must be finite numbers.")

    train_data = data.iloc[:-test_size]
    test_data = data.iloc[-test_size:]

    last_training_value = float(train_data["Actual"].iloc[-1])

    actual_values = test_data["Actual"].to_numpy(dtype=float)
    predicted_values = np.full(
        shape=test_size,
        fill_value=last_training_value,
        dtype=float,
    )

    errors = actual_values - predicted_values

    results = pd.DataFrame({
        "Date": test_data["Date"].to_numpy(),
        "Actual": actual_values,
        "Predicted": predicted_values,
        "Error": errors,
        "Absolute Error": np.abs(errors),
    })

    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "results": results,
    }


def compare_forecasting_models(time_series, test_size=1):
    """Compare linear-trend and last-value baseline forecasts."""
    linear_result = evaluate_linear_trend(
        time_series,
        test_size=test_size,
    )

    baseline_result = evaluate_last_value_baseline(
        time_series,
        test_size=test_size,
    )

    comparison = pd.DataFrame([
        {
            "Model": "Linear Trend",
            "MAE": linear_result["MAE"],
            "RMSE": linear_result["RMSE"],
        },
        {
            "Model": "Last-Value Baseline",
            "MAE": baseline_result["MAE"],
            "RMSE": baseline_result["RMSE"],
        },
    ])

    return {
        "comparison": comparison,
        "linear_results": linear_result["results"],
        "baseline_results": baseline_result["results"],
    }

    
