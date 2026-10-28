
from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


def load_file(uploaded_file):
    """
    Load a CSV or Excel file into a Pandas DataFrame.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        ValueError: If the file is unsupported, empty, or invalid.
    """
    if uploaded_file is None:
        raise ValueError("Please upload a file.")

    filename = getattr(uploaded_file, "name", "")
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Please upload a CSV or XLSX file."
        )

    try:
        uploaded_file.seek(0)

        if extension == ".csv":
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file, engine="openpyxl")

    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded CSV file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            "The CSV file could not be parsed. Check its encoding and format."
        ) from exc
    except (ValueError, ImportError, OSError) as exc:
        raise ValueError(
            "The file could not be read. Check that it is a valid CSV or XLSX file."
        ) from exc

    if data.empty:
        raise ValueError(
            "The file contains no data rows. Please upload a dataset with records."
        )

    if len(data.columns) == 0:
        raise ValueError("The file does not contain any columns.")

    if all(str(column).startswith("Unnamed:") for column in data.columns):
        raise ValueError(
            "No usable column headers were found. Check the first row of the file."
        )

    return data
