import pandas as pd
import numpy as np
import streamlit as st

from src.analytics import calculate_kpis
from src.dynamic_filters import detect_column_types, apply_filters
from src.exports import export_csv, export_excel, generate_text_report
from src.file_loader import load_file
from src.forecasting import (
    prepare_time_series,
    forecast_linear_trend,
    evaluate_linear_trend,
    compare_forecasting_models,
)
from src.insights import generate_insights
from src.profiling import profile_dataframe
from src.validation import validate_dataframe, get_quality_summary
from src.visualizations import get_chart_columns, create_chart


def parse_dates_flexibly(values: pd.Series, column_name: str = "") -> pd.Series:
    """Parse common date formats, including mixed text formats and timestamps."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce")

    # Numeric values are ambiguous (they may be sales, IDs, or Excel dates).
    # Interpret them as Excel serial dates only when the column name suggests a date.
    name_suggests_date = any(
        word in str(column_name).lower()
        for word in ("date", "time", "day", "timestamp")
    )
    if pd.api.types.is_numeric_dtype(values):
        if name_suggests_date:
            numeric = pd.to_numeric(values, errors="coerce")
            # Excel's common 1900 date system; reject implausible serials.
            serials = numeric.where(numeric.between(1, 100000))
            return pd.to_datetime(
                serials, unit="D", origin="1899-12-30", errors="coerce"
            )
        return pd.to_datetime(pd.Series(pd.NaT, index=values.index), errors="coerce")

    # pandas >= 2 supports format="mixed"; the fallback handles older versions.
    candidates = []
    for dayfirst in (False, True):
        try:
            parsed = pd.to_datetime(
                values, format="mixed", dayfirst=dayfirst, errors="coerce"
            )
        except (TypeError, ValueError):
            try:
                parsed = pd.to_datetime(values, dayfirst=dayfirst, errors="coerce")
            except (TypeError, ValueError, OverflowError):
                parsed = pd.Series(pd.NaT, index=values.index)
        except OverflowError:
            parsed = pd.Series(pd.NaT, index=values.index)
        candidates.append(parsed)

    # Explicit formats cover common inputs on older pandas versions and strings
    # such as YYYYMMDD, DD.MM.YYYY, and month-name dates.
    common_formats = (
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d",
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
        "%m-%d-%Y", "%m/%d/%Y", "%m.%d.%Y",
        "%d-%m-%y", "%d/%m/%y", "%m/%d/%y",
        "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
    )
    for fmt in common_formats:
        try:
            parsed = pd.to_datetime(values, format=fmt, errors="coerce")
        except (TypeError, ValueError, OverflowError):
            parsed = pd.Series(pd.NaT, index=values.index)
        candidates.append(parsed)

    # Keep the interpretation that successfully parses the most non-null values.
    return max(candidates, key=lambda parsed: int(parsed.notna().sum()))


st.set_page_config(
    page_title="Intelligent Sales Analytics",
    page_icon="📊",
    layout="wide",
)

# Global styling: presentation only; analytics logic remains unchanged.
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1500px; }
      [data-testid="stAppViewContainer"] { background: var(--background-color); }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(128, 128, 128, 0.18); }
      [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { letter-spacing: -0.02em; }
      .dashboard-hero {
        padding: 1.5rem 1.7rem; margin-bottom: 1.2rem; border-radius: 18px;
        border: 1px solid rgba(128, 128, 128, 0.22);
        background: linear-gradient(120deg, rgba(59, 130, 246, 0.13), rgba(16, 185, 129, 0.08));
      }
      .dashboard-eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .76rem;
        font-weight: 700; opacity: .72; margin-bottom: .45rem; }
      .dashboard-title { font-size: clamp(1.65rem, 3vw, 2.35rem); font-weight: 800;
        letter-spacing: -.045em; line-height: 1.12; margin: 0 0 .55rem 0; }
      .dashboard-subtitle { font-size: 1rem; opacity: .82; margin: 0; max-width: 850px; }
      [data-testid="stMetric"] { padding: 1rem 1rem .85rem; border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.22); background: rgba(128, 128, 128, 0.045); }
      [data-testid="stMetricLabel"] { font-weight: 600; }
      [data-testid="stMetricValue"] { letter-spacing: -.035em; }
      button[data-baseweb="tab"] { font-weight: 650; }
      div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .35rem; }
      div[data-testid="stTabs"] [data-baseweb="tab"] { border-radius: 9px 9px 0 0; padding: .65rem 1rem; }
      [data-testid="stFileUploader"] { border-radius: 14px; }
      div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
      .section-note { opacity: .72; font-size: .92rem; }
      @media (max-width: 700px) {
        .dashboard-hero { padding: 1.1rem; }
        div[data-testid="stTabs"] [data-baseweb="tab"] { padding: .45rem .55rem; }
      }
    </style>
    <div class="dashboard-hero">
      <div class="dashboard-eyebrow">Analytics workspace</div>
      <div class="dashboard-title">📊 Intelligent Sales Analytics</div>
      <p class="dashboard-subtitle">Turn CSV and Excel files into clear KPIs, interactive visualizations,
      business insights, and forecast comparisons.</p>
    </div>
    """,
    unsafe_allow_html=True,
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
except (ValueError, UnicodeDecodeError, pd.errors.ParserError, OSError, ImportError) as error:
    st.error(f"Unable to read this file: {error}")
    st.info("Check that the file is a valid CSV or XLSX file and try uploading it again.")
    st.stop()

if not isinstance(data, pd.DataFrame):
    st.error("The uploaded file could not be converted into a table.")
    st.stop()

if data.empty:
    st.warning("The uploaded file contains no data rows. Upload a file with at least one data row.")
    st.stop()

if len(data.columns) == 0:
    st.error("The uploaded file has no columns to analyze.")
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


# Sidebar filters are generated dynamically from the uploaded dataset.
column_types = detect_column_types(data)
filters = {}

with st.sidebar:
    st.header("🎛️ Filter Your Data")
    st.caption("Filters update the dashboard across all tabs.")

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
        numeric_values = numeric_values[np.isfinite(numeric_values)]

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
        dates = parse_dates_flexibly(data[column], column).dropna()

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

overview_tab, explore_tab, insights_tab, forecast_tab, reports_tab = st.tabs([
    "Overview", "Explore Data", "Insights", "Forecasting", "Reports"
])

with overview_tab:
    st.subheader("Sales & Business KPIs")
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



    st.subheader("🧮 Custom Analysis")
    st.caption(
        "Group records by a field, choose an aggregation, and rank the results "
        "to find the highest- or lowest-performing groups."
    )

    analysis_group_columns = list(dict.fromkeys(
        column_types["categorical"] + column_types["date"]
    ))
    analysis_numeric_columns = []
    for column in filtered_data.columns:
        numeric_values = pd.to_numeric(filtered_data[column], errors="coerce").dropna()
        if not numeric_values.empty and np.isfinite(numeric_values).any():
            analysis_numeric_columns.append(column)

    if filtered_data.empty:
        st.info("No rows are available for custom analysis. Adjust your filters.")
    elif not analysis_group_columns:
        st.info("No categorical or date columns are available for Group By.")
    elif not analysis_numeric_columns:
        st.info("No numeric columns are available for aggregation.")
    else:
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)

        with analysis_col1:
            group_column = st.selectbox(
                "Group By",
                options=analysis_group_columns,
                key="custom_analysis_group_by",
            )

        with analysis_col2:
            measure_column = st.selectbox(
                "Measure",
                options=analysis_numeric_columns,
                key="custom_analysis_measure",
            )

        with analysis_col3:
            aggregation_label = st.selectbox(
                "Aggregation",
                options=["Sum", "Average", "Maximum", "Minimum", "Count"],
                key="custom_analysis_aggregation",
            )

        ranking_col1, ranking_col2 = st.columns(2)
        with ranking_col1:
            ranking = st.selectbox(
                "Priority",
                options=["Top (highest first)", "Bottom (lowest first)"],
                key="custom_analysis_priority",
            )
        with ranking_col2:
            top_n = st.number_input(
                "Number of groups (Top N)",
                min_value=1,
                max_value=max(1, int(filtered_data[group_column].nunique(dropna=True))),
                value=min(5, max(1, int(filtered_data[group_column].nunique(dropna=True)))),
                step=1,
                key="custom_analysis_top_n",
            )

        analysis_frame = filtered_data[[group_column, measure_column]].copy()
        analysis_frame[measure_column] = pd.to_numeric(
            analysis_frame[measure_column], errors="coerce"
        )
        analysis_frame = analysis_frame.dropna(subset=[group_column])

        aggregation_functions = {
            "Sum": "sum",
            "Average": "mean",
            "Maximum": "max",
            "Minimum": "min",
            "Count": "count",
        }
        result_column = f"{aggregation_label} of {measure_column}"

        if analysis_frame.empty:
            st.info("There are no valid rows for this analysis.")
        else:
            grouped_result = (
                analysis_frame.groupby(group_column, dropna=False)[measure_column]
                .agg(aggregation_functions[aggregation_label])
                .reset_index(name=result_column)
            )
            grouped_result = grouped_result.dropna(subset=[result_column])

            ascending = ranking.startswith("Bottom")
            grouped_result = grouped_result.sort_values(
                result_column, ascending=ascending
            ).head(int(top_n))

            if grouped_result.empty:
                st.info("No grouped results are available for the selected options.")
            else:
                st.dataframe(grouped_result, use_container_width=True, hide_index=True)
                import plotly.express as px

                analysis_figure = px.bar(
                    grouped_result,
                    x=group_column,
                    y=result_column,
                    title=f"{ranking.split(' ')[0]} {int(top_n)} by {result_column}",
                    labels={group_column: group_column, result_column: result_column},
                )
                st.plotly_chart(analysis_figure, use_container_width=True)


    st.subheader("📈 Interactive Visualizations")

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

        x_column = None
        y_column = None
        aggregation = "Sum"
        chart_type = None

        if not chart_types:
            st.info(
                "No suitable chart is available for this dataset. "
                "Try a dataset with at least one usable numeric column."
            )
        else:
            chart_type = st.selectbox(
                "Choose chart type",
                options=chart_types,
                key="chart_type",
            )

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


        if chart_type is not None:
            try:
                figure = create_chart(
                    data=filtered_data,
                    chart_type=chart_type,
                    x_column=x_column,
                    y_column=y_column,
                    aggregation=aggregation,
                )
                st.plotly_chart(figure, use_container_width=True)
            except (ValueError, TypeError, KeyError) as error:
                st.warning(f"Unable to create this chart: {error}")


with explore_tab:
    st.subheader("🗂️ Data Preview")
    st.caption(f"{len(data):,} records and {len(data.columns):,} columns")
    st.dataframe(data.head(10), use_container_width=True)

    summary = get_quality_summary(data)

    st.subheader("🛡️ Data Quality")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{summary['rows']:,}")
    c2.metric("Columns", f"{summary['columns']:,}")
    c3.metric("Missing Cells", f"{summary['missing_cells']:,}")
    c4.metric("Duplicate Rows", f"{summary['duplicate_rows']:,}")


    st.subheader("🔎 Filtered Dataset")
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


with insights_tab:
    st.subheader("💡 Automated Business Insights")

    insights = generate_insights(filtered_data)

    st.caption(
        "Insights are calculated from the currently filtered dataset. "
        "Review the underlying records before acting on any finding."
    )

    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("There are not enough valid records to generate insights for the current filters.")


with reports_tab:
    st.subheader("📥 Export & Download")

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


with forecast_tab:
    st.subheader("🔮 Sales Forecasting")

    if filtered_data.empty:
        st.info("No data is available for forecasting with the current filters.")
    else:
        # Detect dates in common formats: ISO, slash/dot/dash separated,
        # compact YYYYMMDD, month names, timestamps, and date-named Excel serials.
        forecast_date_columns = []
        for column in filtered_data.columns:
            series = filtered_data[column]
            if not (
                pd.api.types.is_datetime64_any_dtype(series)
                or pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or (
                    pd.api.types.is_numeric_dtype(series)
                    and any(word in str(column).lower() for word in ("date", "time", "day", "timestamp"))
                )
            ):
                continue

            parsed_dates = parse_dates_flexibly(series, column)
            valid_original = int(series.notna().sum())
            valid_dates = int(parsed_dates.notna().sum())
            date_name_hint = any(
                word in str(column).lower()
                for word in ("date", "time", "day", "timestamp")
            )
            if valid_dates >= 3 and valid_original > 0 and (
                valid_dates / valid_original >= 0.8 or date_name_hint
            ):
                forecast_date_columns.append(column)

        # Numeric-looking text columns can be forecast too, after conversion.
        forecast_numeric_columns = []
        for column in filtered_data.columns:
            numeric_values = pd.to_numeric(filtered_data[column], errors="coerce")
            numeric_values = numeric_values.where(np.isfinite(numeric_values))
            valid_original = int(filtered_data[column].notna().sum())
            valid_numeric = int(numeric_values.notna().sum())
            if valid_numeric > 0 and (
                pd.api.types.is_numeric_dtype(filtered_data[column])
                or (valid_original > 0 and valid_numeric / valid_original >= 0.8)
            ):
                forecast_numeric_columns.append(column)

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
                    forecast_source = filtered_data.copy()
                    forecast_source[selected_date_column] = parse_dates_flexibly(
                        forecast_source[selected_date_column], selected_date_column
                    )
                    forecast_source[selected_value_column] = pd.to_numeric(
                        forecast_source[selected_value_column], errors="coerce"
                    )
                    forecast_source[selected_value_column] = forecast_source[
                        selected_value_column
                    ].where(np.isfinite(forecast_source[selected_value_column]))
                    forecast_source = forecast_source.dropna(
                        subset=[selected_date_column, selected_value_column]
                    )

                    time_series = prepare_time_series(
                        forecast_source,
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

                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    st.warning(f"Forecast could not be generated for these selections: {error}")


