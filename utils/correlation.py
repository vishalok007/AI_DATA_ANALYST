import pandas as pd


def calculate_correlation(df, max_columns=25):
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return None

    if numeric_df.shape[1] > max_columns:
        variances = numeric_df.var(numeric_only=True).sort_values(ascending=False)
        top_cols = variances.head(max_columns).index.tolist()
        numeric_df = numeric_df[top_cols]

    return numeric_df.corr()
