
import pandas as pd

from src.validation import validate_dataframe, get_quality_summary


def test_missing_values_and_duplicates():
    data = pd.DataFrame({
        "Product": ["Laptop", "Laptop", "Mouse"],
        "Sales": [100, 100, None],
    })

    errors, warnings = validate_dataframe(data)

    assert errors == []
    assert any("missing" in warning.lower() for warning in warnings)
    assert any("duplicate" in warning.lower() for warning in warnings)


def test_quality_summary():
    data = pd.DataFrame({
        "Product": ["Laptop", "Mouse"],
        "Sales": [100, None],
    })

    summary = get_quality_summary(data)

    assert summary["rows"] == 2
    assert summary["columns"] == 2
    assert summary["missing_cells"] == 1
    assert summary["duplicate_rows"] == 0


def test_empty_dataframe():
    data = pd.DataFrame()

    errors, warnings = validate_dataframe(data)

    assert len(errors) > 0
