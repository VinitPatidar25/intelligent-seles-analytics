
import pandas as pd

from src.insights import generate_insights, find_date_column


def test_empty_data_returns_message():
    result = generate_insights(pd.DataFrame())

    assert len(result) == 1
    assert "No insights" in result[0]


def test_sales_and_profit_insights():
    data = pd.DataFrame({
        "Region": ["North", "South", "North"],
        "Sales": [100, 200, 150],
        "Profit": [20, 40, 10],
    })

    result = generate_insights(data)
    text = " ".join(result)

    assert "450.00" in text
    assert "70.00" in text
    assert "North" in text


def test_missing_values_are_reported():
    data = pd.DataFrame({
        "Product": ["Laptop", "Mouse"],
        "Sales": [100, None],
    })

    result = generate_insights(data)

    assert any("missing" in item.lower() for item in result)


def test_dataset_without_sales_does_not_invent_sales_insight():
    data = pd.DataFrame({
        "Product": ["Laptop", "Mouse"],
        "Quantity": [2, 3],
    })

    result = generate_insights(data)

    assert not any("Total Sales is" in item for item in result)
    assert not any("Total Revenue is" in item for item in result)


def test_date_column_detection():
    data = pd.DataFrame({
        "Order Date": [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ],
        "Sales": [100, 200, 150],
    })

    assert find_date_column(data) == "Order Date"


def test_outlier_is_flagged():
    data = pd.DataFrame({
        "Quantity": [10, 11, 10, 12, 11, 100],
    })

    result = generate_insights(data)

    assert any("outlier" in item.lower() for item in result)


def test_empty_filtered_dataset():
    data = pd.DataFrame(columns=["Region", "Sales"])

    result = generate_insights(data)

    assert any("No insights" in item for item in result)


def test_insights_use_only_supplied_rows():
    data = pd.DataFrame({
        "Region": ["North", "South"],
        "Sales": [100, 200],
    })

    result = generate_insights(data)
    text = " ".join(result)

    assert "300.00" in text
