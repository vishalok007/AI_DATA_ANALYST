import pandas as pd

def calculate_statistics(df, profile, max_columns=40):
    numeric_columns = profile["numeric_columns"]

    if not numeric_columns:
        return pd.DataFrame()

    selected_columns = numeric_columns[:max_columns]

    statistics = df[selected_columns].describe().T
    statistics["variance"] = df[selected_columns].var()
    statistics["skewness"] = df[selected_columns].skew()
    statistics["kurtosis"] = df[selected_columns].kurt()

    return statistics.round(2)
