import pandas as pd
from utils.statistics import calculate_statistics
from utils.correlation import calculate_correlation


def test_calculate_statistics():
    df = pd.DataFrame({
        "num1": [1, 2, 3, 4, 5],
        "num2": [10, 20, 30, 40, 50],
        "text": ["a", "b", "c", "d", "e"],
    })
    profile = {"numeric_columns": ["num1", "num2"]}

    stats = calculate_statistics(df, profile)

    assert "mean" in stats.columns
    assert "variance" in stats.columns
    assert "skewness" in stats.columns
    assert stats.shape[0] == 2


def test_calculate_correlation():
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [2, 4, 6, 8],
    })

    corr = calculate_correlation(df)
    assert round(corr.loc["A", "B"], 2) == 1.0
