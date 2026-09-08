"""
clean_text.py
-------------
Function to normalize/clean text and categorical columns.
"""

import pandas as pd


def clean_text(df):
    """
    Normalize text/categorical columns so that values like
    ' Male ', 'MALE', and 'male' are all treated the same way.
    Numeric columns are left untouched.
    """
    df = df.copy()

    # Only touch columns that hold text/categorical data (object or category dtype)
    text_columns = df.select_dtypes(include=["object", "category", "string"]).columns

    for col in text_columns:
        # Convert to string first so we can safely use .str methods,
        # but keep actual NaNs as NaN (don't turn them into the text "nan")
        df[col] = df[col].apply(
            lambda x: str(x).strip().lower() if pd.notna(x) else x
        )

    print(f"Cleaned text formatting in columns: {list(text_columns)}")
    return df
