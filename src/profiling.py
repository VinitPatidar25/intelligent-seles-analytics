
import pandas as pd


def profile_dataframe(data):
    """
    Create a summary of each column in the dataset.
    """
    profile = pd.DataFrame({
        "Column": data.columns.astype(str),
        "Data Type": [str(dtype) for dtype in data.dtypes],
        "Missing Values": data.isna().sum().to_numpy(),
        "Missing (%)": (
            data.isna().mean().mul(100).round(2).to_numpy()
        ),
        "Unique Values": data.nunique(dropna=True).to_numpy(),
    })

    return profile
