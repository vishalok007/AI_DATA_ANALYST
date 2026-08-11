import numpy as np
import pandas as pd
from utils.cleaning import (
    convert_dtype,
    drop_columns,
    drop_missing_values,
    fill_constant,
    fill_mean,
    fill_median,
    fill_mode,
    remove_duplicates,
)


def test_remove_duplicates():
    df = pd.DataFrame({"A": [1, 1, 2], "B": ["a", "a", "b"]})
    cleaned, removed = remove_duplicates(df)
    assert len(cleaned) == 2
    assert removed == 1


def test_drop_missing_values():
    df = pd.DataFrame({"A": [1, np.nan, 3], "B": ["x", "y", "z"]})
    cleaned, removed = drop_missing_values(df)
    assert len(cleaned) == 2
    assert removed == 1


def test_fill_mean():
    df = pd.DataFrame({"A": [10.0, 20.0, np.nan]})
    cleaned = fill_mean(df, "A")
    assert cleaned["A"].iloc[2] == 15.0


def test_fill_median():
    df = pd.DataFrame({"A": [10.0, 20.0, 30.0, np.nan]})
    cleaned = fill_median(df, "A")
    assert cleaned["A"].iloc[3] == 20.0


def test_fill_mode():
    df = pd.DataFrame({"A": ["apple", "apple", "banana", None]})
    cleaned = fill_mode(df, "A")
    assert cleaned["A"].iloc[3] == "apple"


def test_fill_constant():
    df = pd.DataFrame({"A": [1, 2, None]})
    cleaned = fill_constant(df, "A", 999)
    assert cleaned["A"].iloc[2] == 999


def test_drop_columns():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    cleaned = drop_columns(df, ["B", "C"])
    assert list(cleaned.columns) == ["A"]


def test_convert_dtype():
    df = pd.DataFrame({"num_str": ["10", "20"], "date_str": ["2025-01-01", "2025-01-02"]})

    c1, err1 = convert_dtype(df, "num_str", "Numeric")
    assert err1 is None
    assert pd.api.types.is_numeric_dtype(c1["num_str"])

    c2, err2 = convert_dtype(df, "date_str", "Datetime")
    assert err2 is None
    assert pd.api.types.is_datetime64_any_dtype(c2["date_str"])
