"""Assemble a Markdown report combining the profile, the cleaning audit
log, charts, and an optional AI narrative into one file a human can
actually read -- the point of glance is to replace the manual version
of this, not just expose its pieces separately.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from glance.charts import categorical_bar_charts, missing_values_chart, numeric_histograms
from glance.cleaning import CleaningReport, clean
from glance.narrate import Narrator
from glance.profiling import ColumnProfile, DatasetProfile, profile


def generate(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    missing_strategy: str = "median_mode",
    narrator: Narrator | None = None,
) -> Path:
    """Run the full pipeline against df and write report.md (plus any
    charts it references) into output_dir. Returns the report's path.

    The missing-values chart reflects df as given -- it's meant to show
    what was wrong before cleaning. The distribution charts run against
    the cleaned data, since that's what someone will actually go on to
    analyze.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_profile = profile(df)
    cleaned_df, cleaning_report = clean(df, missing_strategy=missing_strategy)
    cleaned_profile = profile(cleaned_df)

    missing_chart = missing_values_chart(raw_profile, output_dir / "missing_values.png")
    histogram_chart = numeric_histograms(
        cleaned_df, cleaned_profile, output_dir / "numeric_histograms.png"
    )
    categorical_chart = categorical_bar_charts(
        cleaned_df, cleaned_profile, output_dir / "categorical_bars.png"
    )

    narrative = None
    if narrator is not None and narrator.enabled:
        narrative = narrator.narrate(cleaned_profile, cleaning_report)

    markdown = _render(
        raw_profile,
        cleaned_profile,
        cleaning_report,
        narrative=narrative,
        missing_chart=missing_chart,
        histogram_chart=histogram_chart,
        categorical_chart=categorical_chart,
    )

    report_path = output_dir / "report.md"
    report_path.write_text(markdown)
    return report_path


def _render(
    raw_profile: DatasetProfile,
    cleaned_profile: DatasetProfile,
    cleaning_report: CleaningReport,
    *,
    narrative: str | None,
    missing_chart: Path | None,
    histogram_chart: Path | None,
    categorical_chart: Path | None,
) -> str:
    lines = ["# glance report", ""]

    if narrative:
        lines += ["## Summary", "", narrative, ""]

    lines += [
        "## Overview",
        "",
        f"- Rows: {raw_profile.row_count}",
        f"- Columns: {raw_profile.column_count}",
        f"- Duplicate rows found: {raw_profile.duplicate_row_count}",
        "",
    ]

    if missing_chart is not None:
        lines += ["## Missing values", "", f"![Missing values]({missing_chart.name})", ""]

    lines += ["## Cleaning", "", _cleaning_table(cleaning_report), ""]

    lines += ["## Columns (after cleaning)", "", _column_table(cleaned_profile.columns), ""]

    if histogram_chart is not None:
        lines += [
            "## Numeric distributions",
            "",
            f"![Numeric distributions]({histogram_chart.name})",
            "",
        ]

    if categorical_chart is not None:
        lines += [
            "## Categorical value counts",
            "",
            f"![Categorical value counts]({categorical_chart.name})",
            "",
        ]

    return "\n".join(lines)


def _cleaning_table(report: CleaningReport) -> str:
    rows = [f"| {step.name} | {step.description} |" for step in report.steps]
    return "\n".join(["| Step | What changed |", "| --- | --- |", *rows])


def _column_table(columns: list[ColumnProfile]) -> str:
    header = ["| Column | Type | Missing | Unique | Detail |", "| --- | --- | --- | --- | --- |"]
    rows = [
        f"| {col.name} | {col.dtype} | {col.missing_pct}% | "
        f"{col.unique_count} | {_column_detail(col)} |"
        for col in columns
    ]
    return "\n".join(header + rows)


def _column_detail(col: ColumnProfile) -> str:
    if col.is_numeric:
        return (
            f"mean={col.mean}, median={col.median}, min={col.min}, "
            f"max={col.max}, outliers={col.outlier_count}"
        )
    if col.top_values:
        top = ", ".join(f"{value!r}: {count}" for value, count in col.top_values[:3])
        return f"top: {top}"
    return "-"
