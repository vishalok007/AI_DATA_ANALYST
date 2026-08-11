import numpy as np
import pandas as pd
from utils.quality import analyze_quality


def test_analyze_quality():
    df = pd.DataFrame({
        "col_a": [1, 2, 2, np.nan],
        "col_b": ["constant", "constant", "constant", "constant"],
        "col_c": [10, 20, 20, 40],
    })

    report = analyze_quality(df)

    assert report["rows"] == 4
    assert report["columns"] == 3
    assert report["missing_values"] == 1
    assert report["duplicate_rows"] == 1
    assert "col_a" in report["missing_columns"]
    assert "col_b" in report["constant_columns"]
    assert isinstance(report["quality_score"], int)
