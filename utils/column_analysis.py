import pandas as pd
def analyze_column(df, column):
    series = df[column]
    analysis = {
        "name": column,
        "dtype": str(series.dtype),
        "missing": int(series.isna().sum()),
        "unique": int(series.nunique()),
        "memory": round(series.memory_usage(deep=True) / 1024, 2)
    }

    # Numeric column analysis
    if pd.api.types.is_numeric_dtype(series):
        analysis.update({
            "mean": round(series.mean(), 2),
            "median": round(series.median(), 2),
            "mode": series.mode().iloc[0] if not series.mode().empty else None,
            "minimum": round(series.min(), 2),
            "maximum": round(series.max(), 2),
            "std": round(series.std(), 2),
            "variance": round(series.var(), 2),
            "skewness": round(series.skew(), 2),
            "kurtosis": round(series.kurt(), 2)
        })
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[
            (series < lower_bound) |
            (series > upper_bound)
        ]
        analysis.update({
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "outlier_count": len(outliers),
            "outliers": outliers.tolist()
            })        
    return analysis