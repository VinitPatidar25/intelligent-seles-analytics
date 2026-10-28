
import pandas as pd

from src.analytics import find_column


def find_date_column(data):
    """Find a usable date column without assuming a fixed column name."""
    hints = ("date", "day", "timestamp", "datetime")

    for column in data.columns:
        series = data[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            if series.notna().any():
                return column

        name_suggests_date = any(
            hint in str(column).lower() for hint in hints
        )

        if name_suggests_date and series.notna().any():
            parsed = pd.to_datetime(series, errors="coerce")
            valid_ratio = parsed.notna().sum() / series.notna().sum()

            if valid_ratio >= 0.8:
                return column

    return None


def generate_insights(data):
    """
    Generate evidence-based business insights from the supplied data.
    The supplied DataFrame may already be filtered.
    """
    insights = []

    if data.empty:
        return ["No insights are available because no rows match the filters."]

    insights.append(
        f"The current analysis contains {len(data):,} records "
        f"and {len(data.columns):,} columns."
    )

    # Data-quality insight
    missing_cells = int(data.isna().sum().sum())

    if missing_cells:
        insights.append(
            f"{missing_cells:,} cells contain missing values. "
            "Review these before making business decisions."
        )
    else:
        insights.append("No missing cell values were found in this dataset.")

    # Supported sales and profit summaries
    sales_column = find_column(
        data, ["sales", "revenue", "total sales"]
    )
    profit_column = find_column(
        data, ["profit", "net profit"]
    )

    if sales_column is not None:
        sales = pd.to_numeric(data[sales_column], errors="coerce")
        valid_sales = sales.dropna()

        if not valid_sales.empty:
            insights.append(
                f"Total {sales_column} is "
                f"{valid_sales.sum():,.2f}, based on "
                f"{len(valid_sales):,} valid numeric records."
            )

    if profit_column is not None:
        profit = pd.to_numeric(data[profit_column], errors="coerce")
        valid_profit = profit.dropna()

        if not valid_profit.empty:
            insights.append(
                f"Total {profit_column} is "
                f"{valid_profit.sum():,.2f}, based on "
                f"{len(valid_profit):,} valid numeric records."
            )

    # Top category by total sales/revenue
    if sales_column is not None:
        category_columns = [
            column for column in data.columns
            if (
                column != sales_column
                and not pd.api.types.is_numeric_dtype(data[column])
                and not pd.api.types.is_datetime64_any_dtype(data[column])
                and not pd.api.types.is_bool_dtype(data[column])
            )
        ]

        if category_columns:
            category_column = category_columns[0]
            temp = data[[category_column, sales_column]].copy()
            temp[sales_column] = pd.to_numeric(
                temp[sales_column], errors="coerce"
            )
            temp = temp.dropna(subset=[category_column, sales_column])

            if not temp.empty:
                totals = temp.groupby(
                    category_column, dropna=True
                )[sales_column].sum()

                if not totals.empty:
                    top_category = totals.idxmax()
                    top_value = totals.max()

                    insights.append(
                        f"{top_category} has the highest total "
                        f"{sales_column} among values in "
                        f"{category_column}: {top_value:,.2f}."
                    )

    # Compare earlier and later halves of the observed date range
    date_column = find_date_column(data)

    if date_column is not None and sales_column is not None:
        temp = data[[date_column, sales_column]].copy()
        temp[date_column] = pd.to_datetime(
            temp[date_column], errors="coerce"
        )
        temp[sales_column] = pd.to_numeric(
            temp[sales_column], errors="coerce"
        )
        temp = temp.dropna(subset=[date_column, sales_column])
        temp = temp.sort_values(date_column)

        if len(temp) >= 2:
            midpoint = len(temp) // 2
            earlier_total = temp.iloc[:midpoint][sales_column].sum()
            later_total = temp.iloc[midpoint:][sales_column].sum()

            if earlier_total != 0:
                change = (
                    (later_total - earlier_total)
                    / abs(earlier_total)
                    * 100
                )

                direction = "increased" if change > 0 else "decreased"
                if change == 0:
                    insights.append(
                        "Sales totals were unchanged between the two "
                        "chronological halves of the available records."
                    )
                else:
                    insights.append(
                        f"The sales total for the later half of records "
                        f"{direction} by {abs(change):.2f}% compared with "
                        "the earlier half. This compares record halves, "
                        "not necessarily equal-length time periods."
                    )
            else:
                insights.append(
                    "A percentage change between the two record halves "
                    "could not be calculated because the earlier sales "
                    "total is zero."
                )

    # Outlier detection using the interquartile range (IQR)
    numeric_columns = [
        column
        for column in data.columns
        if pd.api.types.is_numeric_dtype(data[column])
        and not pd.api.types.is_bool_dtype(data[column])
    ]

    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="coerce").dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = values[
            (values < lower_bound) | (values > upper_bound)
        ]

        if not outliers.empty:
            insights.append(
                f"{len(outliers):,} possible outlier(s) were detected "
                f"in {column} using the IQR method. Investigate these "
                "values; an outlier is not automatically an error."
            )

    return insights
