import pandas as pd
import pytest

from glance.charts import categorical_bar_charts, missing_values_chart, numeric_histograms
from glance.profiling import profile


def _is_real_png(path):
    return path.exists() and path.stat().st_size > 0 and path.suffix == ".png"


def test_missing_values_chart_written_when_missing_present(tmp_path):
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    prof = profile(df)

    out = missing_values_chart(prof, tmp_path / "missing.png")

    assert out is not None
    assert _is_real_png(out)


def test_missing_values_chart_returns_none_when_nothing_missing(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    prof = profile(df)

    out = missing_values_chart(prof, tmp_path / "missing.png")

    assert out is None
    assert not (tmp_path / "missing.png").exists()


def test_numeric_histograms_written_for_numeric_columns(tmp_path):
    df = pd.DataFrame({"amount": [1, 2, 3, 4, 5], "label": ["a", "b", "c", "d", "e"]})
    prof = profile(df)

    out = numeric_histograms(df, prof, tmp_path / "hist.png")

    assert out is not None
    assert _is_real_png(out)


def test_numeric_histograms_returns_none_with_no_numeric_columns(tmp_path):
    df = pd.DataFrame({"label": ["a", "b", "c"]})
    prof = profile(df)

    out = numeric_histograms(df, prof, tmp_path / "hist.png")

    assert out is None


def test_categorical_bar_charts_written_for_categorical_columns(tmp_path):
    df = pd.DataFrame({"category": ["a", "a", "b", "c", "c", "c"]})
    prof = profile(df)

    out = categorical_bar_charts(df, prof, tmp_path / "bars.png")

    assert out is not None
    assert _is_real_png(out)


def test_categorical_bar_charts_returns_none_with_no_categorical_columns(tmp_path):
    df = pd.DataFrame({"amount": [1, 2, 3]})
    prof = profile(df)

    out = categorical_bar_charts(df, prof, tmp_path / "bars.png")

    assert out is None


@pytest.mark.parametrize("n_numeric", [1, 2, 3, 4, 7])
def test_numeric_histograms_grid_handles_various_column_counts(tmp_path, n_numeric):
    # Exercises the grid-sizing/ceil-division math across counts that do
    # and don't evenly divide the row width -- this is exactly the kind
    # of off-by-one that's easy to get wrong and easy to verify: it
    # should never raise regardless of how many columns there are.
    df = pd.DataFrame({f"col_{i}": [1, 2, 3] for i in range(n_numeric)})
    prof = profile(df)

    out = numeric_histograms(df, prof, tmp_path / f"hist_{n_numeric}.png")

    assert _is_real_png(out)
