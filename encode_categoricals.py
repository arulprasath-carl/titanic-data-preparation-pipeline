"""
encode_categoricals.py
-----------------------
Function to encode categorical/text columns into numeric form.
"""


def encode_categoricals(df):
    """
    Convert all remaining categorical/text columns into numeric form:
      - Binary mapping for two-category columns (e.g. sex).
      - Ordinal encoding for naturally ordered columns (e.g. pclass
        is already numeric/ordinal, so no change needed there).
      - One-hot encoding for nominal (unordered) categorical columns
        (e.g. embarked), using drop_first=True to avoid redundant
        dummy columns.
    """
    import pandas as pd

    df = df.copy()

    # Binary mapping: sex has exactly two categories.
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"male": 0, "female": 1})

    # Binary mapping: 'alone' is a boolean column (True/False).
    if "alone" in df.columns:
        df["alone"] = df["alone"].astype(int)

    # pclass (1st, 2nd, 3rd class) is already a numeric ordinal
    # column, so it naturally represents ordinal ranking without
    # further encoding.

    # One-hot encode remaining nominal categorical columns.
    nominal_columns = [col for col in df.select_dtypes(include=["object", "category", "string"]).columns]

    if nominal_columns:
        df = pd.get_dummies(df, columns=nominal_columns, drop_first=True)

    # Make sure any boolean columns created by get_dummies become
    # proper 0/1 integers so the final dataset is fully numeric.
    bool_columns = df.select_dtypes(include=["bool"]).columns
    df[bool_columns] = df[bool_columns].astype(int)

    print(f"Encoded categorical columns. Remaining nominal columns one-hot encoded: {nominal_columns}")
    return df
