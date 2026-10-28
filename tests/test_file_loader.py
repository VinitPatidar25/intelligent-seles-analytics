
from io import BytesIO

import pandas as pd
import pytest

from src.file_loader import load_file


def test_load_csv():
    file = BytesIO(b"Product,Sales\nLaptop,100\nMouse,20\n")
    file.name = "sales.csv"

    data = load_file(file)

    assert len(data) == 2
    assert list(data.columns) == ["Product", "Sales"]


def test_load_excel():
    file = BytesIO()

    with pd.ExcelWriter(file, engine="openpyxl") as writer:
        pd.DataFrame({
            "Product": ["Laptop"],
            "Sales": [100],
        }).to_excel(writer, index=False)

    file.seek(0)
    file.name = "sales.xlsx"

    data = load_file(file)

    assert len(data) == 1
    assert data.loc[0, "Sales"] == 100


def test_unsupported_file():
    file = BytesIO(b"test")
    file.name = "sales.txt"

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file(file)


def test_empty_csv():
    file = BytesIO(b"")
    file.name = "empty.csv"

    with pytest.raises(ValueError, match="empty"):
        load_file(file)
