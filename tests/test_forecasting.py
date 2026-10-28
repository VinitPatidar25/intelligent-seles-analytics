import numpy as np
import pandas as pd
import pytest

from src.forecasting import evaluate_linear_trend

from src.forecasting import prepare_time_series
from src.forecasting import forecast_linear_trend

@pytest.fixture
def transaction_data():
    return pd.DataFrame({
        "Order Date": [
            "2026-01-02",
            "2026-01-01",
            "2026-01-01",
            "2026-01-03",
        ],
        "Sales": [200, 100, 50, 150],
    })


def test_dates_are_sorted(transaction_data):
    result = prepare_time_series(
        transaction_data, "Order Date", "Sales"
    )

    assert result["Date"].is_monotonic_increasing


def test_duplicate_dates_are_aggregated(transaction_data):
    result = prepare_time_series(
        transaction_data, "Order Date", "Sales"
    )

    assert len(result) == 3
    assert result.loc[0, "Actual"] == 150


def test_output_has_expected_columns(transaction_data):
    result = prepare_time_series(
        transaction_data, "Order Date", "Sales"
    )

    assert list(result.columns) == ["Date", "Actual"]


def test_invalid_rows_are_removed():
    data = pd.DataFrame({
        "Date": ["2026-01-01", "invalid", "2026-01-03", "2026-01-04"],
        "Sales": [100, 200, "unknown", 300],
    })

    result = prepare_time_series(data, "Date", "Sales")

    assert len(result) == 2
    assert result["Actual"].tolist() == [100, 300]


def test_missing_date_column_raises_error(transaction_data):
    with pytest.raises(ValueError, match="date column"):
        prepare_time_series(
            transaction_data, "Missing Date", "Sales"
        )


def test_missing_numeric_column_raises_error(transaction_data):
    with pytest.raises(ValueError, match="numeric column"):
        prepare_time_series(
            transaction_data, "Order Date", "Missing Sales"
        )


def test_empty_data_raises_error():
    with pytest.raises(ValueError, match="No rows"):
        prepare_time_series(
            pd.DataFrame(columns=["Date", "Sales"]),
            "Date",
            "Sales",
        )


def test_too_few_valid_dates_raises_error():
    data = pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-01"],
        "Sales": [100, 200],
    })

    with pytest.raises(ValueError, match="2 distinct dates"):
        prepare_time_series(data, "Date", "Sales")
 
def test_linear_trend_forecasts_expected_values():
    history = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=4, freq="D"),
        "Actual": [100, 120, 140, 160],
    })

    result = forecast_linear_trend(history, periods=2)

    assert len(result) == 2
    assert result["Forecast"].tolist() == pytest.approx([180, 200])
    assert result["Date"].tolist() == list(
        pd.to_datetime(["2026-01-05", "2026-01-06"])
    )


def test_forecast_has_expected_columns():
    history = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=3, freq="D"),
        "Actual": [10, 20, 30],
    })

    result = forecast_linear_trend(history, periods=3)

    assert list(result.columns) == ["Date", "Forecast"]
    assert len(result) == 3


def test_forecast_rejects_empty_history():
    with pytest.raises(ValueError, match="No historical data"):
        forecast_linear_trend(
            pd.DataFrame(columns=["Date", "Actual"])
        )


def test_forecast_requires_two_observations():
    history = pd.DataFrame({
        "Date": ["2026-01-01"],
        "Actual": [100],
    })

    with pytest.raises(ValueError, match="At least 2"):
        forecast_linear_trend(history)


def test_forecast_rejects_invalid_periods():
    history = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=3, freq="D"),
        "Actual": [10, 20, 30],
    })

    with pytest.raises(ValueError, match="positive integer"):
        forecast_linear_trend(history, periods=0)


def test_forecast_rejects_duplicate_dates():
    history = pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-01", "2026-01-02"],
        "Actual": [10, 20, 30],
    })

    with pytest.raises(ValueError, match="one observation per date"):
        forecast_linear_trend(history)


def test_forecast_rejects_missing_columns():
    history = pd.DataFrame({"Sales": [10, 20, 30]})

    with pytest.raises(ValueError, match="Date and Actual"):
        forecast_linear_trend(history)
def test_evaluate_linear_trend_returns_mae_rmse_and_results():
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "Actual": [10, 20, 30, 40, 50],
    })

    result = evaluate_linear_trend(time_series, test_size=2)

    assert set(result.keys()) == {"MAE", "RMSE", "results"}
    assert result["MAE"] >= 0
    assert result["RMSE"] >= 0
    assert len(result["results"]) == 2
    assert list(result["results"].columns) == [
        "Date", "Actual", "Predicted", "Error", "Absolute Error"
    ]


def test_evaluate_linear_trend_perfect_linear_data():
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "Actual": [10, 20, 30, 40, 50],
    })

    result = evaluate_linear_trend(time_series, test_size=2)

    assert result["MAE"] == pytest.approx(0.0)
    assert result["RMSE"] == pytest.approx(0.0)


@pytest.mark.parametrize("test_size", [0, -1, 4, 2.5, True])
def test_evaluate_linear_trend_rejects_invalid_test_size(test_size):
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "Actual": [10, 20, 30, 40, 50],
    })

    with pytest.raises(ValueError):
        evaluate_linear_trend(time_series, test_size=test_size)


def test_evaluate_linear_trend_requires_three_observations():
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=2, freq="D"),
        "Actual": [10, 20],
    })

    with pytest.raises(ValueError, match="At least 3 observations"):
        evaluate_linear_trend(time_series)


def test_evaluate_linear_trend_rejects_duplicate_dates():
    time_series = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-01-01", "2026-01-02",
            "2026-01-02", "2026-01-04",
        ]),
        "Actual": [10, 20, 30, 40],
    })

    with pytest.raises(ValueError, match="duplicate dates"):
        evaluate_linear_trend(time_series)


def test_evaluate_linear_trend_rejects_missing_columns():
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=3, freq="D"),
        "Sales": [10, 20, 30],
    })

    with pytest.raises(ValueError, match="Date and Actual"):
        evaluate_linear_trend(time_series)


def test_evaluate_linear_trend_rejects_non_numeric_actual_values():
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=3, freq="D"),
        "Actual": [10, "invalid", 30],
    })

    with pytest.raises(ValueError, match="valid numeric"):
        evaluate_linear_trend(time_series)


from src.forecasting import evaluate_last_value_baseline


def test_last_value_baseline_calculates_metrics_correctly():
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "Actual": [10, 20, 30, 40, 50],
    })

    result = evaluate_last_value_baseline(time_series, test_size=2)

    assert result["MAE"] == pytest.approx(15.0)
    assert result["RMSE"] == pytest.approx(np.sqrt(250.0))
    assert result["results"]["Predicted"].tolist() == [30.0, 30.0]
    assert result["results"]["Actual"].tolist() == [40.0, 50.0]


def test_last_value_baseline_handles_constant_sales():
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=4, freq="D"),
        "Actual": [100, 100, 100, 100],
    })

    result = evaluate_last_value_baseline(time_series, test_size=1)

    assert result["MAE"] == pytest.approx(0.0)
    assert result["RMSE"] == pytest.approx(0.0)


def test_last_value_baseline_rejects_too_few_observations():
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=2, freq="D"),
        "Actual": [10, 20],
    })

    with pytest.raises(ValueError, match="At least 3 observations"):
        evaluate_last_value_baseline(time_series)


def test_last_value_baseline_rejects_invalid_test_size():
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "Actual": [10, 20, 30, 40, 50],
    })

    with pytest.raises(ValueError, match="positive integer"):
        evaluate_last_value_baseline(time_series, test_size=0)


def test_last_value_baseline_rejects_duplicate_dates():
    time_series = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-01-01", "2026-01-02",
            "2026-01-02", "2026-01-04",
        ]),
        "Actual": [10, 20, 30, 40],
    })

    with pytest.raises(ValueError, match="duplicate dates"):
        evaluate_last_value_baseline(time_series)

from src.forecasting import compare_forecasting_models


def test_compare_forecasting_models_returns_both_models():
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "Actual": [10, 20, 30, 40, 50],
    })

    result = compare_forecasting_models(time_series, test_size=2)

    assert list(result["comparison"]["Model"]) == [
        "Linear Trend",
        "Last-Value Baseline",
    ]
    assert list(result["comparison"].columns) == [
        "Model", "MAE", "RMSE"
    ]
    assert len(result["linear_results"]) == 2
    assert len(result["baseline_results"]) == 2


def test_linear_trend_beats_baseline_on_linear_data():
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "Actual": [10, 20, 30, 40, 50],
    })

    result = compare_forecasting_models(time_series, test_size=2)
    comparison = result["comparison"].set_index("Model")

    assert comparison.loc["Linear Trend", "MAE"] == pytest.approx(0.0)
    assert comparison.loc["Linear Trend", "RMSE"] == pytest.approx(0.0)
    assert (
        comparison.loc["Linear Trend", "MAE"]
        < comparison.loc["Last-Value Baseline", "MAE"]
    )


def test_model_comparison_rejects_insufficient_data():
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=2, freq="D"),
        "Actual": [10, 20],
    })

    with pytest.raises(ValueError, match="At least 3 observations"):
        compare_forecasting_models(time_series)

def test_model_comparison_handles_constant_sales():
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "Actual": [100, 100, 100, 100, 100],
    })

    result = compare_forecasting_models(time_series, test_size=2)

    assert result["comparison"]["MAE"].notna().all()
    assert result["comparison"]["RMSE"].notna().all()
    assert result["comparison"]["MAE"].tolist() == pytest.approx([0.0, 0.0])
    assert result["comparison"]["RMSE"].tolist() == pytest.approx([0.0, 0.0])


def test_model_comparison_rejects_invalid_numeric_values():
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=4, freq="D"),
        "Actual": [10, 20, float("nan"), 40],
    })

    with pytest.raises(ValueError, match="valid numeric"):
        compare_forecasting_models(time_series, test_size=1)


def test_model_comparison_rejects_duplicate_dates():
    time_series = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-01-01", "2026-01-02",
            "2026-01-02", "2026-01-04",
        ]),
        "Actual": [10, 20, 30, 40],
    })

    with pytest.raises(ValueError, match="duplicate dates"):
        compare_forecasting_models(time_series, test_size=1)


def test_model_comparison_rejects_test_size_too_large():
    time_series = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "Actual": [10, 20, 30, 40, 50],
    })

    with pytest.raises(ValueError, match="Keep at least 2 observations"):
        compare_forecasting_models(time_series, test_size=4)
