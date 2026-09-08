"""
validate_data.py
-----------------
Function to validate that the Titanic dataset has all required columns.
Contains the custom DataValidationError exception.
"""


class DataValidationError(Exception):
    """Raised when the Titanic dataset is missing required columns."""
    pass


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
