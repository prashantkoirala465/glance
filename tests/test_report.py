import pandas as pd

from glance.narrate import Narrator
from glance.report import generate


def _messy_df():
    return pd.DataFrame(
        {
            "Amount": [1.0, 2.0, None, 4.0, 100.0, 1.0],
            "Category": ["a", "b", "b", None, "a", "a"],
        }
    )


def test_generate_writes_report_and_charts(tmp_path):
    report_path = generate(_messy_df(), tmp_path)

    assert report_path == tmp_path / "report.md"
    assert report_path.exists()
    assert (tmp_path / "missing_values.png").exists()
    assert (tmp_path / "numeric_histograms.png").exists()
    assert (tmp_path / "categorical_bars.png").exists()


def test_report_references_charts_by_relative_filename(tmp_path):
    report_path = generate(_messy_df(), tmp_path)
    text = report_path.read_text()

    assert "![Missing values](missing_values.png)" in text
    assert "![Numeric distributions](numeric_histograms.png)" in text
    assert "![Categorical value counts](categorical_bars.png)" in text


def test_report_includes_overview_and_cleaning_and_column_sections(tmp_path):
    report_path = generate(_messy_df(), tmp_path)
    text = report_path.read_text()

    assert "## Overview" in text
    assert "Rows: 6" in text
    assert "## Cleaning" in text
    assert "handle_missing_values" in text
    assert "## Columns (after cleaning)" in text
    assert "amount" in text  # normalized column name
    assert "category" in text


def test_report_has_no_missing_chart_section_when_nothing_missing(tmp_path):
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})
    report_path = generate(df, tmp_path)
    text = report_path.read_text()

    assert not (tmp_path / "missing_values.png").exists()
    assert "Missing values" not in text.split("## Cleaning")[0].replace("## Overview", "")


def test_report_has_no_summary_section_without_narrator(tmp_path):
    report_path = generate(_messy_df(), tmp_path)
    text = report_path.read_text()

    assert "## Summary" not in text


def test_report_has_no_summary_section_with_disabled_narrator(tmp_path):
    report_path = generate(_messy_df(), tmp_path, narrator=Narrator(api_key=None))
    text = report_path.read_text()

    assert "## Summary" not in text


class _FakeMessages:
    def create(self, **_kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="This dataset looks fine overall.")]
        )


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def test_report_includes_summary_section_with_enabled_narrator(tmp_path):
    narrator = Narrator(api_key=None)
    narrator._client = _FakeClient()

    report_path = generate(_messy_df(), tmp_path, narrator=narrator)
    text = report_path.read_text()

    assert "## Summary" in text
    assert "This dataset looks fine overall." in text


def test_generate_creates_output_dir_if_missing(tmp_path):
    output_dir = tmp_path / "nested" / "reports"
    report_path = generate(_messy_df(), output_dir)

    assert report_path.exists()
