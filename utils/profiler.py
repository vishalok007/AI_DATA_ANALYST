import pandas as pd
from utils.datatype import detect_column_types
def profile_dataset(df):
    """Generate a profile for any dataset."""
    profile = {}
    # Basic Information
    profile["rows"] = df.shape[0]
    profile["columns"] = df.shape[1]

    # Missing Values
    profile["missing_values"] = int(df.isnull().sum().sum())

    # Duplicate Rows
    profile["duplicate_rows"] = int(df.duplicated().sum())

    # Memory Usage
    memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    profile["memory_usage"] = f"{memory:.2f} MB"

    # Column Types
    column_types = detect_column_types(df)
    profile["numeric_columns"] = column_types["numeric"]
    profile["categorical_columns"] = column_types["categorical"]
    profile["datetime_columns"] = column_types["datetime"]
    profile["boolean_columns"] = column_types["boolean"]
    profile["identifier_columns"] = column_types["identifier"]
    profile["text_columns"] = column_types["text"]

    return profile