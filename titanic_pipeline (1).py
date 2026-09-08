"""
Titanic Survivor Data Preparation Pipeline
--------------------------------------------
Minor Project: Machine Learning Data Preparation

This script is a DATA PREPARATION pipeline only.
It does NOT train any prediction model (no Logistic Regression,
Random Forest, Decision Tree, etc.). Its job is to take the raw
Titanic dataset and turn it into a clean, encoded, scaled, and
train-test-split dataset that could be handed to ANY downstream
ML model.

Pipeline order:
load_data -> validate_data -> clean_text -> drop_useless ->
handle_missing -> encode_categoricals -> split_data -> scale_features
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------
# Custom Exception
# ------------------------------------------------------------------
class DataValidationError(Exception):
    """Raised when the Titanic dataset is missing required columns."""
    pass


# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
def load_data():
    """
    Load the Titanic dataset directly from Seaborn's built-in datasets.
    Returns:
        pd.DataFrame: the raw Titanic dataset.
    """
    df = sns.load_dataset("titanic")
    return df


# ------------------------------------------------------------------
# 2. VALIDATE DATA
# ------------------------------------------------------------------
def validate_data(df):
    """
    Make sure all the columns we plan to work with actually exist
    in the dataset. If something is missing, we stop early with a
    clear, custom error instead of failing later with a confusing
    KeyError.
    """
    required_columns = [
        "survived", "pclass", "sex", "age", "sibsp", "parch",
        "fare", "embarked", "class", "who", "adult_male",
        "deck", "embark_town", "alone"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise DataValidationError(
            f"Missing required column(s) in dataset: {missing_columns}"
        )

    print("Validation passed: all required columns are present.")
    return df


# ------------------------------------------------------------------
# 3. CLEAN TEXT
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 4. DROP USELESS COLUMNS
# ------------------------------------------------------------------
def drop_useless(df):
    """
    Remove columns that either:
      (a) leak information about the target in an unhelpful/duplicate way,
      (b) are mostly empty, or
      (c) don't add real predictive value (pure identifiers/duplicates).

    We are careful NEVER to drop 'survived' (the target).
    """
    df = df.copy()

    columns_to_drop = []

    # 'deck' is missing for the vast majority of passengers (~77% missing
    # in the raw Titanic dataset). Imputing that much data would be
    # unreliable, so we drop it instead of guessing.
    if "deck" in df.columns:
        missing_ratio = df["deck"].isna().mean()
        if missing_ratio > 0.5:
            columns_to_drop.append("deck")

    # 'embark_town' duplicates the information already captured by
    # 'embarked' (just spelled out as full town names). Keeping both
    # is redundant, so we drop the duplicate version.
    if "embark_town" in df.columns and "embarked" in df.columns:
        columns_to_drop.append("embark_town")

    # 'class' duplicates 'pclass' (same information, one is text,
    # one is numeric). We keep the numeric 'pclass' and drop 'class'.
    if "class" in df.columns and "pclass" in df.columns:
        columns_to_drop.append("class")

    # 'who' (man/woman/child) is largely derivable from 'sex' and 'age',
    # and 'adult_male' is a redundant boolean also derivable from
    # 'sex' and 'age'. Keeping all of them creates duplicate signals.
    if "who" in df.columns:
        columns_to_drop.append("who")
    if "adult_male" in df.columns:
        columns_to_drop.append("adult_male")

    # 'alive' is a text version of the target 'survived' itself.
    # This is a direct target leak, so it must be removed.
    if "alive" in df.columns:
        columns_to_drop.append("alive")

    # Safety check: never drop the target column by accident.
    columns_to_drop = [c for c in columns_to_drop if c != "survived"]

    df = df.drop(columns=columns_to_drop, errors="ignore")

    print(f"Dropped columns: {columns_to_drop}")
    return df


# ------------------------------------------------------------------
# 5. HANDLE MISSING VALUES
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 6. ENCODE CATEGORICAL VARIABLES
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 7. SPLIT DATA
# ------------------------------------------------------------------
def split_data(df, target_column="survived", test_size=0.2, random_state=42):
    """
    Separate features (X) and target (y), then perform a stratified
    80/20 train-test split so that both sets preserve the original
    survived/not-survived ratio.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Data split into train/test sets (test_size={test_size}, stratified on '{target_column}').")
    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------
# 8. SCALE FEATURES
# ------------------------------------------------------------------
def scale_features(X_train, X_test):
    """
    Scale numeric features using StandardScaler.

    IMPORTANT (data leakage prevention):
    The scaler is FIT only on X_train. X_test is only TRANSFORMED
    using the statistics (mean/std) learned from X_train. This
    ensures no information from the test set leaks into training.
    """
    scaler = StandardScaler()

    # Fit only on training data
    X_train_scaled_array = scaler.fit_transform(X_train)

    # Transform test data using the SAME fitted scaler (no re-fitting)
    X_test_scaled_array = scaler.transform(X_test)

    # Preserve feature names by putting the scaled arrays back into DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=X_test.columns, index=X_test.index)

    print("Features scaled using StandardScaler (fit on training data only).")
    return X_train_scaled, X_test_scaled


# ------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("TITANIC SURVIVOR DATA PREPARATION PIPELINE")
    print("=" * 60)

    try:
        # Step 1: Load
        df = load_data()
        print(f"\n[1] Original dataset shape: {df.shape}")

        # Step 2: Validate
        df = validate_data(df)

        # Step 3: Clean text
        df = clean_text(df)

        # Step 4: Drop useless columns
        df = drop_useless(df)

        # Step 5: Handle missing values
        df = handle_missing(df)

        # Step 6: Encode categoricals
        df = encode_categoricals(df)
        print(f"\n[2] Final processed feature shape (including target): {df.shape}")

        # Step 7: Split data
        X_train, X_test, y_train, y_test = split_data(df)
        print(f"\n[3] X_train shape: {X_train.shape}")
        print(f"    X_test shape:  {X_test.shape}")
        print(f"    y_train shape: {y_train.shape}")
        print(f"    y_test shape:  {y_test.shape}")

        print("\n[4] Target distribution (full dataset):")
        print(df["survived"].value_counts(normalize=True).round(3))

        print("\n[5] Target distribution (train set):")
        print(y_train.value_counts(normalize=True).round(3))

        print("\n[6] Target distribution (test set):")
        print(y_test.value_counts(normalize=True).round(3))

        # Step 8: Scale features
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Final scaled X_train shape: {X_train_scaled.shape}")
        print(f"Final scaled X_test shape:  {X_test_scaled.shape}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    except DataValidationError as ve:
        print("\nDATA VALIDATION ERROR:")
        print(ve)

    except Exception as e:
        print("\nUNEXPECTED ERROR DURING PREPROCESSING:")
        print(e)


if __name__ == "__main__":
    main()
