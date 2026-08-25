"""Draft a plain-English summary of a profiled, cleaned dataset using a
local Ollama model. No external provider, no API key, no data leaving
your machine -- narrate() talks to a local Ollama server the same way
the `ollama` CLI's own HTTP client would, using nothing but the
standard library.

Deliberately narrow: the model only ever sees aggregate statistics --
column names, dtypes, counts, summary stats, and what the cleaning
pipeline changed -- never the raw rows.

Optional end to end: construct a Narrator with no model name and
narrate() just returns None. Nothing else in glance depends on this
module running.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from glance.cleaning import CleaningReport
from glance.profiling import ColumnProfile, DatasetProfile

_DEFAULT_HOST = "http://localhost:11434"
_TIMEOUT_SECONDS = 120


class OllamaUnavailableError(RuntimeError):
    """Raised when a model was configured but the local Ollama server
    couldn't be reached, or returned an error (e.g. the model hasn't
    been pulled)."""


class Narrator:
    def __init__(self, model: str | None = None, host: str = _DEFAULT_HOST) -> None:
        self._model = model
        self._host = host.rstrip("/")

    @property
    def enabled(self) -> bool:
        return self._model is not None

    def narrate(self, profile: DatasetProfile, cleaning_report: CleaningReport) -> str | None:
        """Return a short plain-English summary, or None if no model
        was configured. Raises OllamaUnavailableError if a model *was*
        configured but the server couldn't be reached -- that's the
        caller's call to make (fail loudly, or fall back silently),
        not something to swallow here.
        """
        if self._model is None:
            return None

        payload = json.dumps(
            {
                "model": self._model,
                "prompt": build_prompt(profile, cleaning_report),
                "stream": False,
            }
        ).encode()
        request = urllib.request.Request(
            f"{self._host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise OllamaUnavailableError(
                f"Ollama at {self._host} returned {exc.code} for model {self._model!r}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OllamaUnavailableError(
                f"couldn't reach Ollama at {self._host} -- is it running? ({exc.reason})"
            ) from exc

        return body["response"].strip()


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
