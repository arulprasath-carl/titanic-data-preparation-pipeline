"""
handle_missing.py
------------------
Function to fill missing values (numeric -> median, categorical -> mode).
"""

import numpy as np


def handle_missing(df):
    """
    Fill missing values:
      - Numeric columns  -> filled with the column median.
      - Categorical/text -> filled with the column mode.

    Note: At this stage we are still working with the full dataset
    before the train/test split. For a stricter, leak-proof pipeline,
    imputation values could instead be learned strictly from
    X_train and then applied to X_test. Here we impute before the
    split for simplicity, since these are structural/demographic
    columns (e.g. age, embarked) whose median/mode is stable and
    not derived from the target.
    """
    df = df.copy()

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=["object", "category", "string"]).columns

    for col in numeric_columns:
        if df[col].isna().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    for col in categorical_columns:
        if df[col].isna().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)

    print("Missing values handled (numeric -> median, categorical -> mode).")
    return df
