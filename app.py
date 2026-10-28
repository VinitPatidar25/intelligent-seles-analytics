import pandas as pd
from src.forecasting import prepare_time_series, forecast_linear_trend
from src.insights import generate_insights
from src.visualizations import get_chart_columns, create_chart
import streamlit as st
from src.exports import export_csv, export_excel, generate_text_report
from src.analytics import calculate_kpis
from src.insights import generate_insights
from src.forecasting import (
    prepare_time_series,
    forecast_linear_trend,
    evaluate_linear_trend,
    compare_forecasting_models,
)
from src.forecasting import (
    prepare_time_series,
    forecast_linear_trend,
    evaluate_linear_trend,
)

from src.file_loader import load_file
from src.validation import validate_dataframe, get_quality_summary
from src.profiling import profile_dataframe
from src.dynamic_filters import detect_column_types, apply_filters
from src.analytics import calculate_kpis


st.set_page_config(
    page_title="Intelligent Sales Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Intelligent Sales Analytics Dashboard")
st.write(
    "Upload a dataset to explore its data quality, "
    "apply dynamic filters, and calculate supported KPIs."
)

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"],
)

if uploaded_file is None:
    st.info("Upload a CSV or XLSX file to begin.")
    st.stop()

try:
    data = load_file(uploaded_file)
except ValueError as error:
    st.error(str(error))
    st.stop()

errors, warnings = validate_dataframe(data)

if errors:
    st.error("Please correct these issues before analysis:")
    for error in errors:
        st.write(f"- {error}")
    st.stop()

if warnings:
    with st.expander("Data-quality warnings", expanded=True):
        for warning in warnings:
            st.warning(warning)

st.subheader("1. Data Preview")
st.caption(f"{len(data):,} records and {len(data.columns):,} columns")
st.dataframe(data.head(10), use_container_width=True)

summary = get_quality_summary(data)

st.subheader("2. Data Quality")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{summary['rows']:,}")
c2.metric("Columns", f"{summary['columns']:,}")
c3.metric("Missing Cells", f"{summary['missing_cells']:,}")
c4.metric("Duplicate Rows", f"{summary['duplicate_rows']:,}")

st.subheader("3. Dynamic Filters")
column_types = detect_column_types(data)
filters = {}

with st.sidebar:
    st.header("Filter Your Data")

    for column in column_types["categorical"]:
        options = data[column].dropna().unique().tolist()

        # Avoid errors if values contain mixed data types.
        options = sorted(options, key=lambda value: str(value))

        selected = st.multiselect(
            f"{column}",
            options=options,
            default=[],
            key=f"category_{column}",
        )

        filters[column] = {
            "type": "category",
            "values": selected,
        }

    for column in column_types["numeric"]:
        values = data[column]
        numeric_values = pd.to_numeric(
           values, errors="coerce"
        ).dropna()

        if numeric_values.empty:
            continue

        minimum = float(numeric_values.min())
        maximum = float(numeric_values.max())

        if minimum < maximum:
            selected_range = st.slider(
                f"{column} range",
                min_value=minimum,
                max_value=maximum,
                value=(minimum, maximum),
                key=f"numeric_{column}",
            )
        else:
            st.caption(f"{column}: all valid values are {minimum:g}")
            selected_range = (minimum, maximum)

        
        if selected_range != (minimum, maximum):
               filters[column] = {
                    "type": "numeric",
                    "min": selected_range[0],
                    "max": selected_range[1],
                   }
        

    for column in column_types["date"]:
        dates = pd.to_datetime(data[column], errors="coerce").dropna()

        if dates.empty:
            continue

        start_date = dates.min().date()
        end_date = dates.max().date()

        selected_dates = st.date_input(
            f"{column} date range",
            value=(start_date, end_date),
            min_value=start_date,
            max_value=end_date,
            key=f"date_{column}",
        )

        if isinstance(selected_dates, (tuple, list)):
            selected_start = selected_dates[0] if len(selected_dates) >= 1 else None
            selected_end = selected_dates[1] if len(selected_dates) >= 2 else None
        else:
            selected_start = selected_dates
            selected_end = selected_dates

        if (
    selected_start != start_date
    or selected_end != end_date):
          filters[column] = {
                "type": "date",
                "start": selected_start,
                "end": selected_end,
            }

filtered_data = apply_filters(data, filters)

st.subheader("4. Sales & Business KPIs")
kpis = calculate_kpis(filtered_data)

# Show only KPIs supported by this dataset.
metric_items = [
    ("Records", "row_count", "{:,.0f}"),
    ("Total Sales", "sales_total", "{:,.2f}"),
    ("Average Sales", "sales_average", "{:,.2f}"),
    ("Total Profit", "profit_total", "{:,.2f}"),
    ("Total Quantity", "quantity_total", "{:,.2f}"),
    ("Profit Margin", "profit_margin_percent", "{:,.2f}%"),
]

available_metrics = [
    (label, key, fmt)
    for label, key, fmt in metric_items
    if key in kpis
]

if available_metrics:
    for start in range(0, len(available_metrics), 3):
        columns = st.columns(3)
        for col, (label, key, fmt) in zip(
            columns, available_metrics[start:start + 3]
        ):
            value = kpis[key]
            display = "N/A" if value is None else fmt.format(value)
            col.metric(label, display)
else:
    st.info("No supported business KPIs were found in this dataset.")

st.subheader("5. Filtered Dataset")
st.caption(
    f"{len(filtered_data):,} of {len(data):,} rows match the selected filters."
)
st.dataframe(filtered_data, use_container_width=True)

with st.expander("Column Profile"):
    st.dataframe(
        profile_dataframe(data),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("6. Interactive Visualizations")

if filtered_data.empty:
    st.info("No data matches the selected filters. Adjust your filters to view charts.")
else:
    chart_columns = get_chart_columns(filtered_data)

    numeric_columns = chart_columns["numeric"]
    categorical_columns = chart_columns["categorical"]
    date_columns = chart_columns["date"]

    chart_types = []

    if numeric_columns:
      chart_types.append("Histogram")

    if categorical_columns and numeric_columns:
        chart_types.append("Bar")

    if date_columns and numeric_columns:
        chart_types.append("Line")

    if len(numeric_columns) >= 2:
        chart_types.append("Scatter")

    chart_type = st.selectbox(
        "Choose chart type",
        options=chart_types,
        key="chart_type",
    )

    x_column = None
    y_column = None
    aggregation = "Sum"
    if not chart_types:
     st.info(
        "No suitable chart is available. "
        "This dataset needs at least one usable numeric column."
    )
    else:
    # Keep the chart selector and chart-building code indented
    # inside this else block.

        if chart_type == "Histogram":
            x_column = st.selectbox(
                "Numeric column",
                options=numeric_columns,
                key="histogram_column",
            )

        elif chart_type == "Bar":
            x_column = st.selectbox(
                "Category column",
                options=categorical_columns,
                key="bar_category",
            )
            y_column = st.selectbox(
                "Measure",
                options=numeric_columns,
                key="bar_measure",
            )
            aggregation = st.selectbox(
                "Aggregation",
                options=["Sum", "Average"],
                key="bar_aggregation",
            )

        elif chart_type == "Line":
            x_column = st.selectbox(
                "Date column",
                options=date_columns,
                key="line_date",
            )
            y_column = st.selectbox(
                "Measure",
                options=numeric_columns,
                key="line_measure",
            )
            aggregation = st.selectbox(
                "Aggregation",
                options=["Sum", "Average"],
                key="line_aggregation",
            )

        elif chart_type == "Scatter":
            x_column = st.selectbox(
                "X-axis numeric column",
                options=numeric_columns,
                key="scatter_x",
            )
            y_options = [
                column for column in numeric_columns
                if column != x_column
            ]
            y_column = st.selectbox(
                "Y-axis numeric column",
                options=y_options,
                key="scatter_y",
            )


    try:
        figure = create_chart(
            data=filtered_data,
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            aggregation=aggregation,
        )
        st.plotly_chart(figure, use_container_width=True)

    except ValueError as error:
        st.warning(str(error))

st.subheader("7. Automated Business Insights")

insights = generate_insights(filtered_data)

st.caption(
    "Insights are calculated from the currently filtered dataset. "
    "Review the underlying records before acting on any finding."
)

for insight in insights:
    st.markdown(f"- {insight}")

st.subheader("8. Export & Download")

if filtered_data.empty:
    st.warning("No rows match the current filters. There is nothing to export.")
else:
    st.caption(
        "Exports contain only the rows in the currently filtered dataset."
    )

    col1, col2 = st.columns(2)

    with col1:
        csv_bytes = export_csv(filtered_data)

        st.download_button(
            label="Download filtered CSV",
            data=csv_bytes,
            file_name="filtered_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        excel_bytes = export_excel(filtered_data)

        st.download_button(
            label="Download filtered Excel",
            data=excel_bytes,
            file_name="filtered_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    report_kpis = calculate_kpis(filtered_data)
    report_insights = generate_insights(filtered_data)

    report_bytes = generate_text_report(
        filtered_data,
        report_kpis,
        report_insights,
    )

    st.download_button(
        label="Download analytics report",
        data=report_bytes,
        file_name="analytics_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

st.subheader("9. Sales Forecasting")

if filtered_data.empty:
    st.info("No data is available for forecasting with the current filters.")
else:
    forecast_columns = get_chart_columns(filtered_data)
    forecast_date_columns = forecast_columns["date"]
    forecast_numeric_columns = forecast_columns["numeric"]

    if not forecast_date_columns:
        st.info(
            "No suitable date column was detected. "
            "Use a date column such as Order Date or Transaction Date."
        )
    elif not forecast_numeric_columns:
        st.info(
            "No numeric column is available for forecasting."
        )
    else:
        forecast_col1, forecast_col2 = st.columns(2)

        with forecast_col1:
            selected_date_column = st.selectbox(
                "Forecast date column",
                options=forecast_date_columns,
                key="forecast_date_column",
            )

        with forecast_col2:
            selected_value_column = st.selectbox(
                "Measure to forecast",
                options=forecast_numeric_columns,
                key="forecast_value_column",
            )

        forecast_periods = st.slider(
            "Number of future days",
            min_value=1,
            max_value=30,
            value=7,
            key="forecast_periods",
        )

        if st.button("Generate Forecast", key="generate_forecast"):
            try:
                time_series = prepare_time_series(
                    filtered_data,
                    selected_date_column,
                    selected_value_column,
                )

                forecast_data = forecast_linear_trend(
                    time_series,
                    periods=forecast_periods,
                )

                historical_trace = time_series.rename(
                    columns={"Actual": "Value"}
                ).copy()
                historical_trace["Type"] = "Historical"

                forecast_trace = forecast_data.rename(
                    columns={"Forecast": "Value"}
                ).copy()
                forecast_trace["Type"] = "Forecast"

                import plotly.express as px

                figure = px.line(
                    historical_trace,
                    x="Date",
                    y="Value",
                    title=f"{selected_value_column}: Historical Data",
                    labels={"Value": selected_value_column},
                )

                figure.add_scatter(
                    x=forecast_data["Date"],
                    y=forecast_data["Forecast"],
                    mode="lines+markers",
                    name="Forecast",
                    line={"dash": "dash"},
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True,
                )

                st.subheader("Predicted Values")
                st.dataframe(
                    forecast_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("Forecast Accuracy Evaluation")

                try:
                    evaluation = evaluate_linear_trend(
                        time_series,
                        test_size=min(7, len(time_series) - 2),
                    )

                    metric_col1, metric_col2 = st.columns(2)

                    with metric_col1:
                        st.metric(
                            "MAE",
                            f"{evaluation['MAE']:.2f}",
                        )

                    with metric_col2:
                        st.metric(
                            "RMSE",
                            f"{evaluation['RMSE']:.2f}",
                        )

                    st.caption(
                        "Lower MAE and RMSE generally indicate better "
                        "predictions. Evaluation uses held-out historical "
                        "observations, not future forecast values."
                    )

                    st.subheader("Actual vs Predicted")
                    st.dataframe(
                        evaluation["results"],
                        use_container_width=True,
                        hide_index=True,
                    )

                except ValueError as error:
                    st.info(
                        f"Accuracy evaluation is unavailable: {error}"
                    )
                
                st.subheader("Model Comparison")

                comparison_result = compare_forecasting_models(
                    time_series,
                    test_size=min(7, len(time_series) - 2),
                )

                st.dataframe(
                    comparison_result["comparison"],
                    use_container_width=True,
                    hide_index=True,
                )

                st.bar_chart(
                    comparison_result["comparison"].set_index("Model")[
                        ["MAE", "RMSE"]
                    ]
                )

                st.caption(
                    "Compare both models on the same held-out observations. "
                    "Lower MAE and RMSE indicate smaller prediction errors. "
                    "Results may vary with the dataset and test period."
                )
    


                st.caption(
                    "Forecasts are estimates based on a linear trend. "
                    "They assume daily steps and do not model seasonality. "
                    "Use them as a baseline, not a guarantee."
                )

            except ValueError as error:
                st.warning(str(error))


