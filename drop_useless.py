"""
drop_useless.py
---------------
Function to drop columns that leak the target or are mostly empty.
"""


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
