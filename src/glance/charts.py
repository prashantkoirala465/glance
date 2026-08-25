"""Generate charts from a profiled DataFrame: a missing-data overview,
numeric distributions, and categorical top-value bar charts.

Each function returns the path it wrote, or None if there was nothing
meaningful to chart (e.g. no column has any missing values) -- callers
shouldn't have to special-case "the dataset happened to be complete" as
an error.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display available in a CLI/CI/server context

import matplotlib.pyplot as plt
import pandas as pd

from glance.profiling import DatasetProfile

_DPI = 120


def missing_values_chart(profile: DatasetProfile, output_path: str | Path) -> Path | None:
    """A horizontal bar chart of missing-value percentage per column."""
    columns_with_missing = sorted(
        (c for c in profile.columns if c.missing_count > 0), key=lambda c: c.missing_pct
    )
    if not columns_with_missing:
        return None

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, max(2, len(columns_with_missing) * 0.4)))
    ax.barh(
        [c.name for c in columns_with_missing],
        [c.missing_pct for c in columns_with_missing],
        color="#c0392b",
    )
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing values by column")
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)
    return output_path


def numeric_histograms(
    df: pd.DataFrame, profile: DatasetProfile, output_path: str | Path
) -> Path | None:
    """A grid of histograms, one per numeric column."""
    numeric_cols = [c.name for c in profile.columns if c.is_numeric]
    if not numeric_cols:
        return None

    output_path = Path(output_path)
    fig, axes = _grid_figure(len(numeric_cols), cols_per_row=3, subplot_size=(4, 3))

    for ax, col in zip(axes, numeric_cols, strict=True):
        ax.hist(df[col].dropna(), bins=20, color="#2980b9")
        ax.set_title(col, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)
    return output_path


def categorical_bar_charts(
    df: pd.DataFrame, profile: DatasetProfile, output_path: str | Path
) -> Path | None:
    """A grid of bar charts, one per categorical column, showing its
    most frequent values (as already computed by profile())."""
    cat_columns = [c for c in profile.columns if not c.is_numeric and c.top_values]
    if not cat_columns:
        return None

    output_path = Path(output_path)
    fig, axes = _grid_figure(len(cat_columns), cols_per_row=2, subplot_size=(5, 3))

    for ax, col in zip(axes, cat_columns, strict=True):
        labels = [str(value) for value, _ in col.top_values]
        counts = [count for _, count in col.top_values]
        ax.bar(labels, counts, color="#27ae60")
        ax.set_title(col.name, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)
    return output_path


def _grid_figure(n_plots: int, *, cols_per_row: int, subplot_size: tuple[int, int]):
    """A figure with n_plots subplots arranged in a grid. Returns (fig,
    axes_to_use) where axes_to_use has exactly n_plots entries in
    creation order; any leftover grid cells (when n_plots doesn't evenly
    fill the last row) are hidden rather than left as empty axes.
    """
    ncols = min(cols_per_row, n_plots)
    nrows = -(-n_plots // ncols)  # ceil division

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(subplot_size[0] * ncols, subplot_size[1] * nrows), squeeze=False
    )
    flat_axes = axes.flatten()
    for ax in flat_axes[n_plots:]:
        ax.axis("off")

    return fig, flat_axes[:n_plots]
