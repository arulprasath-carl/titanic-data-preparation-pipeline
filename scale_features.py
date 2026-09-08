"""
scale_features.py
-------------------
Function to scale features using StandardScaler (fit on train only).
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


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
