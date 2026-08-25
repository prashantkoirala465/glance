"""A transparent, auditable cleaning pipeline.

Every transformation is a named step that reports exactly what it did and
why -- cleaning a dataset should never be a black box between "messy" and
"clean". Nothing here silently drops information a report doesn't
mention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

MISSING_STRATEGIES = ("median_mode", "drop")


@dataclass
class CleanStepResult:
    name: str
    description: str


@dataclass
class CleaningReport:
    steps: list[CleanStepResult] = field(default_factory=list)


def clean(
    df: pd.DataFrame, *, missing_strategy: str = "median_mode"
) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the standard cleaning pipeline against df, returning a cleaned
    copy and a report of exactly what changed.

    missing_strategy:
      - "median_mode" (default): numeric columns filled with their
        median, categorical columns filled with their mode.
      - "drop": rows with any missing value are dropped entirely.
    """
    if missing_strategy not in MISSING_STRATEGIES:
        raise ValueError(
            f"unknown missing_strategy {missing_strategy!r}, expected one of {MISSING_STRATEGIES}"
        )

    df = df.copy()
    report = CleaningReport()

    df, step = _normalize_column_names(df)
    report.steps.append(step)

    df, step = _strip_whitespace(df)
    report.steps.append(step)

    df, step = _drop_duplicate_rows(df)
    report.steps.append(step)

    df, step = _handle_missing_values(df, strategy=missing_strategy)
    report.steps.append(step)

    return df, report


def _normalize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, CleanStepResult]:
    original = list(df.columns)
    normalized = [str(c).strip().lower().replace(" ", "_") for c in original]

    # Two differently-named columns ("Total Sales", "total_sales") can
    # normalize to the same name -- pandas allows duplicate column
    # labels, but nearly everything downstream breaks in confusing ways
    # if that happens silently. Disambiguate instead.
    seen: dict[str, int] = {}
    deduped = []
    collisions = 0
    for name in normalized:
        if name in seen:
            seen[name] += 1
            collisions += 1
            deduped.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            deduped.append(name)

    df.columns = deduped
    changed = sum(1 for o, n in zip(original, deduped, strict=True) if o != n)

    description = (
        f"renamed {changed} column(s) to lowercase snake_case"
        if changed
        else "column names already clean"
    )
    if collisions:
        description += f"; {collisions} name collision(s) disambiguated with a numeric suffix"
    return df, CleanStepResult(name="normalize_column_names", description=description)


def _strip_whitespace(df: pd.DataFrame) -> tuple[pd.DataFrame, CleanStepResult]:
    changed_cells = 0
    touched_cols = 0

    for col in df.select_dtypes(include="object").columns:
        col_changed = 0

        def _strip(value: object) -> object:
            nonlocal col_changed
            if isinstance(value, str):
                stripped = value.strip()
                if stripped != value:
                    col_changed += 1
                return stripped
            # Anything that isn't a string -- NaN, or a stray non-string
            # value in a mixed-type column -- passes through untouched.
            # pandas' own .str accessor would silently turn non-string
            # cells into NaN here, which is worse than doing nothing.
            return value

        df[col] = df[col].map(_strip)
        if col_changed:
            touched_cols += 1
            changed_cells += col_changed

    description = (
        f"stripped leading/trailing whitespace from {changed_cells} cell(s) "
        f"across {touched_cols} text column(s)"
        if changed_cells
        else "no whitespace to strip"
    )
    return df, CleanStepResult(name="strip_whitespace", description=description)


def _drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, CleanStepResult]:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(df)

    description = (
        f"dropped {dropped} exact duplicate row(s)" if dropped else "no duplicate rows found"
    )
    return df, CleanStepResult(name="drop_duplicate_rows", description=description)


def _handle_missing_values(
    df: pd.DataFrame, *, strategy: str
) -> tuple[pd.DataFrame, CleanStepResult]:
    if strategy == "drop":
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped = before - len(df)
        description = (
            f"dropped {dropped} row(s) containing any missing value"
            if dropped
            else "no missing values found"
        )
        return df, CleanStepResult(name="handle_missing_values", description=description)

    filled_summary = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            if pd.isna(fill_value):
                filled_summary.append(
                    f"'{col}' ({missing} missing, but no non-missing value to fill from)"
                )
                continue
            df[col] = df[col].fillna(fill_value)
            filled_summary.append(f"'{col}' ({missing} filled with median {fill_value:.2f})")
        else:
            mode = df[col].mode(dropna=True)
            if mode.empty:
                filled_summary.append(
                    f"'{col}' ({missing} missing, but no non-missing value to fill from)"
                )
                continue
            fill_value = mode.iloc[0]
            df[col] = df[col].fillna(fill_value)
            filled_summary.append(f"'{col}' ({missing} filled with mode {fill_value!r})")

    description = (
        "filled missing values in " + "; ".join(filled_summary)
        if filled_summary
        else "no missing values found"
    )
    return df, CleanStepResult(name="handle_missing_values", description=description)
