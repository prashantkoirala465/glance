# glance

Point it at a messy CSV or Excel file. Get a cleaned dataset, a column
profile, a handful of charts, and — optionally — a plain-English
summary, all written out as a single Markdown report.

It's the first 30 minutes of any data project — load it, see what's
missing, see what's weird, clean the obvious stuff, look at the
distributions — done once, consistently, instead of re-typed by hand
at the top of every new notebook.

## Install

```bash
pip install glance
```

Excel support is an optional extra:

```bash
pip install "glance[excel]"    # .xlsx / .xls
```

`--narrate` needs [Ollama](https://ollama.com) running locally — no
extra Python dependency, no API key, nothing leaves your machine:

```bash
brew install ollama        # or see ollama.com for other platforms
ollama pull llama3.2       # any model works; smaller ones are faster
```

## Quickstart

```bash
glance data.csv
```

This writes `data_glance/report.md` (plus the charts it references)
next to your input file. Point `-o` somewhere else if you'd rather:

```bash
glance data.csv -o ./reports/latest
```

Try it against the sample dataset in this repo — a small sales export
with the usual mess baked in on purpose (a duplicate row, inconsistent
whitespace, missing values, one clear outlier):

```bash
glance examples/messy_sales.csv
```

Add `--narrate` with a model name for a short AI-written summary
paragraph at the top of the report (needs Ollama running locally, see
above):

```bash
glance data.csv --narrate llama3.2
```

If Ollama isn't running, or the model isn't pulled, glance prints a
clear warning and still writes the rest of the report — narration is
the one optional piece, everything else works regardless. Point at a
non-default Ollama install with `--ollama-host` or the `OLLAMA_HOST`
environment variable.

Missing values are filled with the column's median/mode by default.
Pass `--missing-strategy drop` to drop incomplete rows instead.

## Library usage

Every CLI step is a plain function you can call directly:

```python
from glance.loading import load
from glance.profiling import profile
from glance.cleaning import clean
from glance.charts import missing_values_chart

df = load("data.csv")
raw_profile = profile(df)

cleaned_df, cleaning_report = clean(df)
for step in cleaning_report.steps:
    print(step.name, "-", step.description)

missing_values_chart(raw_profile, "missing.png")
```

## What it actually does

1. **Load** (`glance.loading`) — reads `.csv`, `.tsv`, `.xlsx`, `.xls`
   into a DataFrame.
2. **Profile** (`glance.profiling`) — per-column dtype, missing %,
   unique count, summary statistics, and IQR-based outlier counts
   (Tukey's rule: outside 1.5× the interquartile range).
3. **Clean** (`glance.cleaning`) — four named, independently-testable
   steps: normalize column names, strip whitespace, drop exact
   duplicate rows, handle missing values. Every step reports exactly
   what it changed — nothing is silently dropped or "fixed" without
   showing up in the audit log. A numeric column with no non-missing
   values to fill from is reported as such rather than claiming a fill
   that didn't happen.
4. **Chart** (`glance.charts`) — a missing-values overview, numeric
   histograms, and categorical bar charts, skipped individually when
   there's nothing meaningful to show (e.g. no missing data, no
   numeric columns).
5. **Narrate** (`glance.narrate`, optional) — one call to a local
   Ollama model that turns the profile and cleaning report into a
   short paragraph. Only aggregate statistics are ever sent — column
   names, dtypes, percentages, summary stats, cleaning step
   descriptions — and since the model runs on your machine, nothing
   leaves it at all. With no model name given this is a no-op; nothing
   else depends on it.
6. **Report** (`glance.report`) — assembles all of the above into one
   `report.md`, with charts referenced by relative filename so the
   output directory stays portable if you move or zip it.

## Development

```bash
pip install -e ".[dev,excel]"
ruff check .
ruff format --check .
pytest
```

## License

MIT — see [LICENSE](LICENSE).
