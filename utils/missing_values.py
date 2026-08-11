import pandas as pd
def analyze_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]    
    if missing.empty:
        return None
    result = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values,
        "Percentage": (
            missing.values / len(df) * 100
        ).round(2)
    })
    return result.sort_values(
        by="Missing Values",
        ascending=False
    )