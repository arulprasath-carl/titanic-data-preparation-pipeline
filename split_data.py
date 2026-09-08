"""
split_data.py
-------------
Function to separate features/target and perform a stratified train-test split.
"""

from sklearn.model_selection import train_test_split


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
