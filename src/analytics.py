
import pandas as pd


def find_column(data, possible_names):
    """
    Find a column using case-insensitive exact name matching.
    """
    lookup = {
        str(column).strip().lower(): column
        for column in data.columns
    }

    for name in possible_names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def calculate_kpis(data):
    """
    Calculate KPIs supported by the available columns.
    No missing business metrics are invented.
    """
    kpis = {
        "row_count": len(data),
        "column_count": len(data.columns),
    }

    sales_column = find_column(
        data, ["sales", "revenue", "total sales"]
    )
    profit_column = find_column(
        data, ["profit", "net profit"]
    )
    quantity_column = find_column(
        data, ["quantity", "qty", "units sold"]
    )

    if sales_column is not None:
        sales = pd.to_numeric(
            data[sales_column], errors="coerce"
        )
        kpis["sales_total"] = sales.sum()
        kpis["sales_average"] = (
            sales.mean() if sales.notna().any() else None
        )

    if profit_column is not None:
        profit = pd.to_numeric(
            data[profit_column], errors="coerce"
        )
        kpis["profit_total"] = profit.sum()
        kpis["profit_average"] = (
            profit.mean() if profit.notna().any() else None
        )

    if quantity_column is not None:
        quantity = pd.to_numeric(
            data[quantity_column], errors="coerce"
        )
        kpis["quantity_total"] = quantity.sum()

    # Profit margin is meaningful only when both sales and profit exist.
    if sales_column is not None and profit_column is not None:
        sales = pd.to_numeric(
            data[sales_column], errors="coerce"
        )
        profit = pd.to_numeric(
            data[profit_column], errors="coerce"
        )

        valid = sales.notna() & profit.notna()
        total_sales = sales[valid].sum()
        total_profit = profit[valid].sum()

        kpis["profit_margin_percent"] = (
            total_profit / total_sales * 100
            if total_sales != 0
            else None
        )

    return kpis
