
import pandas as pd
import pytest

from src.visualizations import get_chart_columns, create_chart


@pytest.fixture
def sales_data():
    return pd.DataFrame({
        "Order Date": [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ],
        "Region": ["North", "South", "North"],
        "Sales": [100, 200, 50],
        "Profit": [20, 40, 10],
    })


def test_detect_chart_columns(sales_data):
    columns = get_chart_columns(sales_data)

    assert "Sales" in columns["numeric"]
    assert "Profit" in columns["numeric"]
    assert "Region" in columns["categorical"]
    assert "Order Date" in columns["date"]


def test_bar_chart_aggregates_category_values(sales_data):
    figure = create_chart(
        sales_data,
        chart_type="Bar",
        x_column="Region",
        y_column="Sales",
        aggregation="Sum",
    )

    plotted_values = list(figure.data[0].y)

    assert sorted(plotted_values) == [150, 200]


def test_line_chart_creates_figure(sales_data):
    figure = create_chart(
        sales_data,
        chart_type="Line",
        x_column="Order Date",
        y_column="Sales",
        aggregation="Sum",
    )

    assert len(figure.data) == 1
    assert len(figure.data[0].y) == 3


def test_histogram_creates_figure(sales_data):
    figure = create_chart(
        sales_data,
        chart_type="Histogram",
        x_column="Sales",
    )

    assert len(figure.data) == 1


def test_scatter_chart_creates_figure(sales_data):
    figure = create_chart(
        sales_data,
        chart_type="Scatter",
        x_column="Sales",
        y_column="Profit",
    )

    assert len(figure.data) == 1
    assert len(figure.data[0].x) == 3


def test_empty_dataset_raises_error():
    with pytest.raises(ValueError, match="No rows"):
        create_chart(
            pd.DataFrame(),
            chart_type="Histogram",
            x_column="Sales",
        )


def test_unknown_chart_type_raises_error(sales_data):
    with pytest.raises(ValueError, match="Unsupported chart type"):
        create_chart(sales_data, chart_type="Pie")
