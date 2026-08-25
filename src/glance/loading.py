"""Load a tabular data file into a DataFrame.

Deliberately thin: pandas already does the hard part. This module's job
is picking the right reader for the file extension and failing with a
clear error when it can't, not reimplementing parsing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class UnsupportedFormatError(ValueError):
    """Raised when the file extension isn't one glance knows how to read."""


_DELIMITED_READERS = {
    ".csv": {},
    ".tsv": {"sep": "\t"},
}
_EXCEL_SUFFIXES = {".xlsx", ".xls"}


def load(path: str | Path) -> pd.DataFrame:
    """Load ``path`` into a DataFrame.

    Supports ``.csv`` and ``.tsv`` with no extra dependencies. ``.xlsx``
    and ``.xls`` need the optional ``excel`` extra (``pip install
    glance[excel]``) -- pandas itself only raises ``ImportError`` deep in
    its own call stack when that's missing, so this catches it and points
    at the fix instead.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")

    suffix = path.suffix.lower()

    if suffix in _DELIMITED_READERS:
        return pd.read_csv(path, **_DELIMITED_READERS[suffix])

    if suffix in _EXCEL_SUFFIXES:
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise UnsupportedFormatError(
                "reading .xlsx/.xls requires the 'excel' extra: pip install glance[excel]"
            ) from exc

    supported = sorted(_DELIMITED_READERS) + sorted(_EXCEL_SUFFIXES)
    raise UnsupportedFormatError(
        f"don't know how to read {suffix!r} files (supported: {', '.join(supported)})"
    )
