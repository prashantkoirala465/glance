import math

import pandas as pd
import pytest

from glance.cleaning import clean


def test_normalize_column_names():
    df = pd.DataFrame({" Full Name ": [1], "AGE": [2], "already_snake": [3]})
    cleaned, report = clean(df)

    assert list(cleaned.columns) == ["full_name", "age", "already_snake"]
    step = next(s for s in report.steps if s.name == "normalize_column_names")
    assert "renamed 2 column" in step.description


def test_normalize_column_names_collision_is_disambiguated():
    df = pd.DataFrame({"Total Sales": [1], "total_sales": [2]})
    cleaned, report = clean(df)

    assert len(set(cleaned.columns)) == len(cleaned.columns)  # no duplicate labels
    step = next(s for s in report.steps if s.name == "normalize_column_names")
    assert "collision" in step.description


def test_strip_whitespace():
    df = pd.DataFrame({"name": ["  Ada  ", "Grace", "  Linus"]})
    cleaned, _ = clean(df, missing_strategy="drop")

    assert list(cleaned["name"]) == ["Ada", "Grace", "Linus"]


def test_strip_whitespace_preserves_non_string_values_in_mixed_column():
    # A mixed-type object column: pandas' own .str accessor would turn
    # the int and the NaN into NaN here. Neither should happen -- only
    # actual strings get touched.
    df = pd.DataFrame({"mixed": [" a ", 42, None]})
    cleaned, _ = clean(df, missing_strategy="drop")

    # missing_strategy="drop" removes the NaN row, leaving the two
    # non-null original rows intact and correctly typed.
    values = list(cleaned["mixed"])
    assert "a" in values
    assert 42 in values
    assert not any(isinstance(v, float) and math.isnan(v) for v in values if not isinstance(v, str))


def test_drop_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    cleaned, report = clean(df)

    assert len(cleaned) == 2
    step = next(s for s in report.steps if s.name == "drop_duplicate_rows")
    assert "dropped 1" in step.description


def test_missing_values_median_mode_strategy():
    df = pd.DataFrame(
        {
            "amount": [10.0, 20.0, None, 30.0],
            "category": ["a", "a", "b", None],
        }
    )
    cleaned, report = clean(df, missing_strategy="median_mode")

    assert cleaned["amount"].isna().sum() == 0
    assert cleaned["amount"].iloc[2] == 20.0  # median of [10, 20, 30]
    assert cleaned["category"].isna().sum() == 0
    assert cleaned["category"].iloc[3] == "a"  # mode of ["a", "a", "b"]

    step = next(s for s in report.steps if s.name == "handle_missing_values")
    assert "amount" in step.description
    assert "category" in step.description


def test_missing_values_drop_strategy():
    df = pd.DataFrame({"amount": [10.0, None, 30.0]})
    cleaned, report = clean(df, missing_strategy="drop")

    assert len(cleaned) == 2
    assert cleaned["amount"].isna().sum() == 0
    step = next(s for s in report.steps if s.name == "handle_missing_values")
    assert "dropped 1 row" in step.description


def test_missing_values_all_missing_column_is_reported_not_silently_filled():
    df = pd.DataFrame({"amount": [None, None], "other": [1, 2]})
    cleaned, report = clean(df, missing_strategy="median_mode")

    # Nothing to compute a median from -- must stay missing, not become NaN-filled-with-NaN.
    assert cleaned["amount"].isna().sum() == 2
    step = next(s for s in report.steps if s.name == "handle_missing_values")
    assert "no non-missing value to fill from" in step.description


def test_invalid_missing_strategy_raises():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="unknown missing_strategy"):
        clean(df, missing_strategy="bogus")


def test_clean_report_has_all_four_steps_in_order():
    df = pd.DataFrame({"a": [1, 2]})
    _, report = clean(df)

    assert [s.name for s in report.steps] == [
        "normalize_column_names",
        "strip_whitespace",
        "drop_duplicate_rows",
        "handle_missing_values",
    ]


def test_clean_does_not_mutate_input():
    df = pd.DataFrame({" A ": [1, 1]})
    original_columns = list(df.columns)

    clean(df)

    assert list(df.columns) == original_columns
