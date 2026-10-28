
from io import BytesIO

import pandas as pd
import pytest

from src.exports import (
    export_csv,
    export_excel,
    generate_text_report,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "Region": ["North", "South"],
        "Sales": [100, 200],
    })


def test_csv_export_contains_expected_rows(sample_data):
    result = export_csv(sample_data)
    restored = pd.read_csv(BytesIO(result))

    assert len(restored) == 2
    assert list(restored.columns) == ["Region", "Sales"]
    assert restored["Sales"].tolist() == [100, 200]


def test_excel_export_contains_expected_rows(sample_data):
    result = export_excel(sample_data)
    restored = pd.read_excel(BytesIO(result), engine="openpyxl")

    assert len(restored) == 2
    assert restored["Region"].tolist() == ["North", "South"]
    assert restored["Sales"].tolist() == [100, 200]


def test_empty_csv_export_raises_error():
    with pytest.raises(ValueError, match="No data"):
        export_csv(pd.DataFrame())


def test_empty_excel_export_raises_error():
    with pytest.raises(ValueError, match="No data"):
        export_excel(pd.DataFrame())


def test_text_report_contains_kpis_and_insights(sample_data):
    kpis = {"row_count": 2, "sales_total": 300}
    insights = ["South has the highest total Sales: 200.00."]

    result = generate_text_report(sample_data, kpis, insights)
    report = result.decode("utf-8")

    assert "INTELLIGENT SALES ANALYTICS REPORT" in report
    assert "Rows analyzed: 2" in report
    assert "Sales Total: 300.00" in report
    assert "South has the highest total Sales" in report


def test_empty_report_raises_error():
    with pytest.raises(ValueError, match="No data"):
        generate_text_report(
            pd.DataFrame(),
            {},
            [],
        )
