"""Draft a plain-English summary of a profiled, cleaned dataset.

Deliberately narrow: the model only ever sees aggregate statistics --
column names, dtypes, counts, summary stats, and what the cleaning
pipeline changed -- never the raw rows. A dataset worth running through
this tool in the first place is exactly the kind that might contain
something sensitive; there's no reason the narration step needs to see
it to describe it.

Optional end to end: construct a Narrator with no API key and narrate()
just returns None. Nothing else in glance depends on this module running.
"""

from __future__ import annotations

from glance.cleaning import CleaningReport
from glance.profiling import ColumnProfile, DatasetProfile

_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 500


class Narrator:
    def __init__(self, api_key: str | None = None) -> None:
        self._client = None
        if api_key:
            # Imported lazily: the anthropic package is an optional
            # extra, and most uses of glance never touch this path.
            from anthropic import Anthropic

            self._client = Anthropic(api_key=api_key)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def narrate(self, profile: DatasetProfile, cleaning_report: CleaningReport) -> str | None:
        """Return a short plain-English summary, or None if no API key
        was configured. Network/API errors are the caller's problem to
        decide how to handle -- they aren't swallowed here, since a
        silent narration failure would be confusing (was there nothing
        to say, or did the call fail?).
        """
        if self._client is None:
            return None

        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": build_prompt(profile, cleaning_report)}],
        )
        return _extract_text(response)


def build_prompt(profile: DatasetProfile, cleaning_report: CleaningReport) -> str:
    lines = [
        "You are summarizing a dataset for someone about to analyze it.",
        "Write a short, plain-English summary (3-5 sentences, one paragraph, "
        "no headings or bullet points) covering: what the dataset looks like, "
        "what stands out (missing data, outliers, imbalanced categories), and "
        "anything about the cleaning that's worth knowing before analyzing it.",
        "",
        f"Rows: {profile.row_count}, Columns: {profile.column_count}, "
        f"Duplicate rows removed during cleaning: {profile.duplicate_row_count}",
        "",
        "Columns:",
    ]
    for col in profile.columns:
        lines.append(f"- {_describe_column(col)}")

    lines.append("")
    lines.append("Cleaning steps applied:")
    for step in cleaning_report.steps:
        lines.append(f"- {step.name}: {step.description}")

    return "\n".join(lines)


def _describe_column(col: ColumnProfile) -> str:
    parts = [f"'{col.name}' ({col.dtype}, {col.missing_pct}% missing, {col.unique_count} unique)"]
    if col.is_numeric:
        parts.append(
            f"mean={col.mean}, median={col.median}, min={col.min}, max={col.max}, "
            f"outliers={col.outlier_count}"
        )
    elif col.top_values:
        top = ", ".join(f"{value!r}: {count}" for value, count in col.top_values)
        parts.append(f"top values: {top}")
    return "; ".join(parts)


def _extract_text(response) -> str:
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""
