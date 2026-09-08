"""
load_data.py
------------
Function to load the Titanic dataset from Seaborn.
"""

import seaborn as sns


def load_data():
    """
    Load the Titanic dataset directly from Seaborn's built-in datasets.
    Returns:
        pd.DataFrame: the raw Titanic dataset.
    """
    df = sns.load_dataset("titanic")
    return df
