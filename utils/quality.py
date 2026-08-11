import pandas as pd
def analyze_quality(df):
    """Analyze dataset quality."""
    report = {}

    # Basic Information
    report["rows"] = df.shape[0]
    report["columns"] = df.shape[1]

    # Missing Values
    report["missing_values"] = int(df.isnull().sum().sum())

    # Duplicate Rows
    report["duplicate_rows"] = int(df.duplicated().sum())

    # Memory Usage
    memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    report["memory_usage"] = f"{memory:.2f} MB"

    # Columns containing missing values
    missing_cols = []
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            missing_cols.append(col)

    report["missing_columns"] = missing_cols

    # Constant Columns
    constant_cols = []

    for col in df.columns:
        if df[col].nunique(dropna=False) == 1:
            constant_cols.append(col)
    report["constant_columns"] = constant_cols

    # High Cardinality Columns
    high_cardinality = []
    for col in df.columns:

        if df[col].dtype == "object":

            if df[col].nunique() > 50:
                high_cardinality.append(col)

    report["high_cardinality_columns"] = high_cardinality

    # Quality Score
    score = 100
    score -= min(report["missing_values"] // 1000, 20)
    score -= min(report["duplicate_rows"], 20)
    score -= len(constant_cols) * 5
    report["quality_score"] = max(score, 0)
    return report