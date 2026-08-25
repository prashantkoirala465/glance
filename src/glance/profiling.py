"""Profile a DataFrame: missing values, dtypes, duplicates, outliers, and
summary statistics -- the mechanical first-look every data project starts
with, done once and consistently instead of re-typed by hand each time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int

    # Numeric columns only; None otherwise.
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    outlier_count: int | None = None

    # Non-numeric columns only; empty otherwise.
    top_values: list[tuple[object, int]] = field(default_factory=list)

    @property
    def is_numeric(self) -> bool:
        return self.mean is not None


@dataclass
class DatasetProfile:
    row_count: int
    column_count: int
    duplicate_row_count: int
    columns: list[ColumnProfile]


def profile(df: pd.DataFrame, top_n: int = 5) -> DatasetProfile:
    """Profile every column in df. top_n controls how many of a
    categorical column's most frequent values get recorded.
    """
    columns = [_profile_column(df[col], top_n=top_n) for col in df.columns]
    return DatasetProfile(
        row_count=len(df),
        column_count=len(df.columns),
        duplicate_row_count=int(df.duplicated().sum()),
        columns=columns,
    )


def _profile_column(series: pd.Series, top_n: int) -> ColumnProfile:
    missing_count = int(series.isna().sum())
    total = len(series)
    missing_pct = (missing_count / total * 100) if total else 0.0

    col = ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        missing_count=missing_count,
        missing_pct=round(missing_pct, 2),
        unique_count=int(series.nunique(dropna=True)),
    )

    if pd.api.types.is_numeric_dtype(series):
        _fill_numeric_stats(col, series.dropna())
    else:
        value_counts = series.dropna().value_counts().head(top_n)
        col.top_values = list(value_counts.items())

    return col


def _fill_numeric_stats(col: ColumnProfile, non_null: pd.Series) -> None:
    if len(non_null) == 0:
        # A numeric column that's entirely missing -- report zeros
        # rather than NaN so downstream formatting doesn't have to
        # special-case this on top of the missing_count already saying
        # exactly this.
        col.mean = col.median = col.std = col.min = col.max = 0.0
        col.outlier_count = 0
        return

    col.mean = round(float(non_null.mean()), 4)
    col.median = round(float(non_null.median()), 4)
    col.std = round(float(non_null.std()), 4) if len(non_null) > 1 else 0.0
    col.min = float(non_null.min())
    col.max = float(non_null.max())
    col.outlier_count = _count_outliers_iqr(non_null)


def _count_outliers_iqr(series: pd.Series) -> int:
    """Count values outside 1.5x the interquartile range -- Tukey's
    classic rule. Not a definitive outlier test, just a fast, well-known
    first signal worth a human's attention.
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())
