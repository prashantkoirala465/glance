import numpy as np
import pandas as pd

from glance.profiling import profile


def _col(prof, name):
    return next(c for c in prof.columns if c.name == name)


def test_profile_basic_shape():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    prof = profile(df)

    assert prof.row_count == 3
    assert prof.column_count == 2
    assert prof.duplicate_row_count == 0


def test_profile_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    prof = profile(df)

    assert prof.duplicate_row_count == 1


def test_profile_missing_values():
    df = pd.DataFrame({"a": [1, None, 3, None]})
    prof = profile(df)
    col = _col(prof, "a")

    assert col.missing_count == 2
    assert col.missing_pct == 50.0


def test_profile_numeric_stats():
    df = pd.DataFrame({"a": [10, 20, 30, 40, 50]})
    col = _col(profile(df), "a")

    assert col.is_numeric
    assert col.mean == 30.0
    assert col.median == 30.0
    assert col.min == 10.0
    assert col.max == 50.0


def test_profile_categorical_top_values():
    df = pd.DataFrame({"category": ["a", "a", "a", "b", "b", "c"]})
    col = _col(profile(df, top_n=2), "category")

    assert not col.is_numeric
    assert col.top_values[0] == ("a", 3)
    assert col.top_values[1] == ("b", 2)
    assert len(col.top_values) == 2  # respects top_n


def test_outlier_detection_iqr():
    # 1-9 are a tight cluster; 1000 is a blatant outlier by any measure.
    df = pd.DataFrame({"a": list(range(1, 10)) + [1000]})
    col = _col(profile(df), "a")

    assert col.outlier_count == 1


def test_no_outliers_in_uniform_data():
    df = pd.DataFrame({"a": [5, 5, 5, 5, 5]})
    col = _col(profile(df), "a")

    # IQR is 0 here -- must not divide by zero or flag everything.
    assert col.outlier_count == 0


def test_profile_entirely_missing_numeric_column():
    df = pd.DataFrame({"a": pd.array([np.nan, np.nan], dtype="float64")})
    col = _col(profile(df), "a")

    assert col.missing_count == 2
    assert col.is_numeric
    assert col.mean == 0.0
    assert col.outlier_count == 0


def test_unique_count_ignores_missing():
    df = pd.DataFrame({"a": [1, 1, None, 2]})
    col = _col(profile(df), "a")

    assert col.unique_count == 2
