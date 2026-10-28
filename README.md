# Intelligent Sales Analytics Dashboard

An interactive Python dashboard for exploring sales data from CSV and
Excel files. It helps users inspect data quality, filter records,
calculate KPIs, visualize trends, generate business insights, and
experiment with simple sales forecasting.

## Features

-   **CSV and Excel uploads** --- load sales data for analysis.
-   **Data preview and profiling** --- inspect records, column types,
    missing values, and duplicate rows.
-   **Dynamic filters** --- filter data using available categorical,
    date, and numeric fields.
-   **Sales KPIs** --- calculate supported metrics from the uploaded
    dataset.
-   **Custom analysis** --- group records and calculate Sum, Average,
    Maximum, Minimum, or Count; rank groups from highest to lowest or
    lowest to highest.
-   **Interactive charts** --- explore the data using supported
    visualizations.
-   **Business insights** --- generate a summary of useful patterns in
    the filtered data.
-   **Exports** --- download supported analysis outputs and filtered
    data.
-   **Forecasting and evaluation** --- explore a simple trend-based
    forecast and compare evaluation metrics such as MAE and RMSE with a
    baseline.
-   **Input and error handling** --- show helpful messages for several
    invalid or empty-data situations.

> Results depend on the columns, quality, and amount of data in the
> uploaded file. Forecasts are exploratory estimates, not guarantees of
> future sales.

## Technology stack

-   Python
-   Streamlit
-   Pandas
-   NumPy
-   Plotly
-   Pytest

## Run locally

### 1. Clone the repository

``` bash
git clone https://github.com/VinitPatidar25/intelligent-seles-analytics.git
cd intelligent-seles-analytics
```

### 2. Create and activate a virtual environment

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the dashboard

``` bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually
`http://localhost:8501`.

## Run tests

From the project root, run:

``` bash
python -m pytest -q
```

The project's existing test suite previously reported **74 passing
tests** during development. Run the command above against the current
checkout to confirm its status.

## Deploy

This project can be deployed using [Streamlit Community
Cloud](https://share.streamlit.io/) by connecting the GitHub repository
and selecting:

-   **Repository:** `VinitPatidar25/intelligent-seles-analytics`
-   **Branch:** `main`
-   **Main file path:** `app.py`

Make sure `requirements.txt` is committed to the repository before
deploying.

**Live demo:** Add your deployed Streamlit URL here once you have copied
it from Streamlit Community Cloud.

## Suggested workflow

1.  Upload a CSV or Excel sales dataset.
2.  Review the data preview and data-quality summary.
3.  Apply the available filters.
4.  Explore KPIs, charts, and custom group-by analysis.
5.  Review insights and, when the data supports it, forecasting results.
6.  Export the results you need.

## Repository

GitHub: https://github.com/VinitPatidar25/intelligent-seles-analytics

## Limitations

-   Available filters, metrics, and charts depend on the uploaded
    dataset's columns and values.
-   Forecasting requires sufficient valid date and numeric observations.
-   Business insights and forecasts should be reviewed against the
    original data before being used for decisions.

## License

No license is specified in this README. Add a license file if you want
to define how others may use, modify, and distribute the project.
