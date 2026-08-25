"""Command-line entry point: point glance at a file and get a Markdown
report back. This is the interface most people will actually use --
the library modules underneath exist so this isn't the *only* one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from glance.cleaning import MISSING_STRATEGIES
from glance.loading import UnsupportedFormatError, load
from glance.narrate import Narrator, OllamaUnavailableError
from glance.report import generate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glance",
        description="Profile, clean, chart, and optionally narrate a messy dataset.",
    )
    parser.add_argument("input", help="path to a .csv, .tsv, .xlsx, or .xls file")
    parser.add_argument(
        "-o",
        "--output",
        help="directory to write the report and charts into (default: <input-stem>_glance)",
    )
    parser.add_argument(
        "--missing-strategy",
        choices=MISSING_STRATEGIES,
        default="median_mode",
        help="how to handle missing values during cleaning (default: median_mode)",
    )
    parser.add_argument(
        "--narrate",
        metavar="MODEL",
        help="add an AI-written summary using a local Ollama model, e.g. --narrate llama3.2 "
        "(requires Ollama running locally: https://ollama.com)",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama server URL for --narrate (default: http://localhost:11434, or $OLLAMA_HOST)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = (
        Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_glance"
    )

    try:
        df = load(input_path)
    except (FileNotFoundError, UnsupportedFormatError) as exc:
        print(f"glance: {exc}", file=sys.stderr)
        return 1

    narrator = Narrator(model=args.narrate, host=args.ollama_host) if args.narrate else None

    try:
        report_path = generate(
            df, output_dir, missing_strategy=args.missing_strategy, narrator=narrator
        )
    except OllamaUnavailableError as exc:
        print(f"glance: {exc}; continuing without a summary", file=sys.stderr)
        report_path = generate(df, output_dir, missing_strategy=args.missing_strategy)

    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
