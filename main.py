"""
main.py
-------
Titanic Survivor Data Preparation Pipeline - Main Entry Point

This file imports the 8 pipeline functions from their separate modules
and runs them in the correct order:

load_data -> validate_data -> clean_text -> drop_useless ->
handle_missing -> encode_categoricals -> split_data -> scale_features

Run this file to execute the full pipeline:
    python main.py
"""

from load_data import load_data
from validate_data import validate_data, DataValidationError
from clean_text import clean_text
from drop_useless import drop_useless
from handle_missing import handle_missing
from encode_categoricals import encode_categoricals
from split_data import split_data
from scale_features import scale_features


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
