
import pandas as pd

from src.dynamic_filters import detect_column_types, apply_filters
from src.analytics import calculate_kpis


def sample_data():
    return pd.DataFrame({
        "Region": ["North", "South", "North", "West"],
        "Sales": [100.0, 200.0, 50.0, None],
        "Profit": [20.0, 40.0, 10.0, 5.0],
        "Quantity": [2, 4, 1, 3],
        "Order Date": [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
        ],
    })


def test_detect_column_types():
    data = sample_data()
    result = detect_column_types(data)

    assert "Region" in result["categorical"]
    assert "Sales" in result["numeric"]
    assert "Quantity" in result["numeric"]
    assert "Order Date" in result["date"]


def test_category_filter():
    data = sample_data()

    result = apply_filters(data, {
        "Region": {
            "type": "category",
            "values": ["North"],
        }
    })

    assert len(result) == 2
    assert result["Region"].eq("North").all()


def test_numeric_filter():
    data = sample_data()

    result = apply_filters(data, {
        "Sales": {
            "type": "numeric",
            "min": 75,
            "max": 250,
        }
    })

    assert len(result) == 2
    assert set(result["Sales"]) == {100, 200}


def test_date_filter():
    data = sample_data()

    result = apply_filters(data, {
        "Order Date": {
            "type": "date",
            "start": "2026-01-02",
            "end": "2026-01-03",
        }
    })

    assert len(result) == 2


def test_empty_category_selection_means_all_rows():
    data = sample_data()

    result = apply_filters(data, {
        "Region": {
            "type": "category",
            "values": [],
        }
    })

    assert len(result) == len(data)


def test_kpis_with_sales_profit_and_quantity():
    data = sample_data()
    kpis = calculate_kpis(data)

    assert kpis["row_count"] == 4
    assert kpis["sales_total"] == 350
    assert kpis["sales_average"] == 350 / 3
    assert kpis["profit_total"] == 75
    assert kpis["quantity_total"] == 10
    assert round(kpis["profit_margin_percent"], 2) == round(
        70 / 350 * 100, 2
    )


def test_kpis_do_not_invent_sales():
    data = pd.DataFrame({
        "Product": ["Laptop", "Mouse"],
        "Quantity": [2, 3],
    })

    kpis = calculate_kpis(data)

    assert kpis["row_count"] == 2
    assert "sales_total" not in kpis
    assert "profit_total" not in kpis
    assert kpis["quantity_total"] == 5


def test_filters_do_not_modify_original_data():
    data = sample_data()
    original = data.copy(deep=True)

    apply_filters(data, {
        "Region": {
            "type": "category",
            "values": ["North"],
        }
    })

    pd.testing.assert_frame_equal(data, original)
