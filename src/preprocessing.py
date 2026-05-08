"""
preprocessing.py
Shared data cleaning and preprocessing utilities.
Imported by all notebooks so logic is defined once.
"""

import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw student performance data.
    - Drops rows with missing target values (score_read, score_math)
    - Drops low-quality columns (ladder, schooldistrict_id)
    - Imputes missing numeric and categorical values
    - Collapses rare ethnicity / degree categories
    """
    df = df.copy()

    # Drop rows missing targets
    df = df.dropna(subset=["score_read", "score_math"])

    # Drop columns not used in modelling
    df = df.drop(columns=["ladder", "schooldistrict_id"], errors="ignore")

    # Numeric imputation
    df["experience"] = df["experience"].fillna(df["experience"].median())

    # Collapse ethnicity to cauc / afam / other
    df["ethnicity"] = df["ethnicity"].apply(
        lambda x: x if x in ["cauc", "afam"] else "other"
    )

    # Categorical imputation
    df = df.fillna(
        {
            "ethnicity": "other",
            "lunch": "Unknown",
            "birth": "Unknown",
            "class_type": "Unknown",
            "t_ethnicity": "Unknown",
            "degree": "Unknown",
        }
    )

    # Collapse degree to bachelor / master / other
    df["degree"] = df["degree"].apply(
        lambda x: x if x in ["bachelor", "master"] else "other"
    )

    return df


def remove_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Remove rows where col is outside 1.5 * IQR."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]


def prepare_data(df_raw: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Full pipeline from raw DataFrame to train/test splits + fitted preprocessor.

    Returns
    -------
    X_train, X_test, y_read_train, y_read_test, y_math_train, y_math_test,
    preprocessor, num_cols, cat_cols
    """
    df = clean_data(df_raw)

    for col in ["score_read", "score_math", "experience"]:
        df = remove_outliers_iqr(df, col)

    X = df.drop(columns=["score_read", "score_math"])
    y_read = df["score_read"]
    y_math = df["score_math"]

    X_train, X_test, y_read_train, y_read_test, y_math_train, y_math_test = (
        train_test_split(X, y_read, y_math, test_size=test_size, random_state=random_state)
    )

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    preprocessor = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    )

    return (
        X_train, X_test,
        y_read_train, y_read_test,
        y_math_train, y_math_test,
        preprocessor, num_cols, cat_cols,
    )
