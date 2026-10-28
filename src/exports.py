
from io import BytesIO
import pandas as pd


def export_csv(data):
    """Convert a DataFrame to downloadable CSV bytes."""
    if data.empty:
        raise ValueError("No data is available to export.")

    return data.to_csv(index=False).encode("utf-8")


def export_excel(data):
    """Convert a DataFrame to downloadable Excel bytes."""
    if data.empty:
        raise ValueError("No data is available to export.")

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        data.to_excel(writer, index=False, sheet_name="Filtered Data")

    return output.getvalue()


def generate_text_report(data, kpis, insights):
    """Create a readable report from the current analysis."""
    if data.empty:
        raise ValueError("No data is available for the report.")

    lines = [
        "INTELLIGENT SALES ANALYTICS REPORT",
        "=" * 38,
        "",
        "DATASET SUMMARY",
        f"Rows analyzed: {len(data)}",
        f"Columns analyzed: {len(data.columns)}",
        f"Missing cells: {int(data.isna().sum().sum())}",
        "",
        "KEY PERFORMANCE INDICATORS",
    ]

    for name, value in kpis.items():
        if value is not None:
            if isinstance(value, (int, float)):
                lines.append(f"{name.replace('_', ' ').title()}: {value:,.2f}")
            else:
                lines.append(f"{name.replace('_', ' ').title()}: {value}")

    lines.extend(["", "AUTOMATED INSIGHTS"])

    for insight in insights:
        lines.append(f"- {insight}")

    lines.extend([
        "",
        "NOTE",
        "This report reflects the dataset currently selected for analysis.",
        "Verify source records before making business decisions.",
    ])

    return "\n".join(lines).encode("utf-8")
