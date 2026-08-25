import pytest

from glance.cleaning import CleaningReport, CleanStepResult
from glance.narrate import Narrator, OllamaUnavailableError, build_prompt
from glance.profiling import ColumnProfile, DatasetProfile

# Kept in sync with conftest.py's _MockOllamaHandler response body.
_MOCK_OLLAMA_REPLY = "This dataset looks mostly clean, with one column worth a second look."


def _sample_profile() -> DatasetProfile:
    return DatasetProfile(
        row_count=10,
        column_count=2,
        duplicate_row_count=1,
        columns=[
            ColumnProfile(
                name="amount",
                dtype="float64",
                missing_count=2,
                missing_pct=20.0,
                unique_count=8,
                mean=42.5,
                median=40.0,
                std=5.1,
                min=10.0,
                max=90.0,
                outlier_count=1,
            ),
            ColumnProfile(
                name="category",
                dtype="object",
                missing_count=0,
                missing_pct=0.0,
                unique_count=3,
                top_values=[("widgets", 6), ("gadgets", 4)],
            ),
        ],
    )


def _sample_cleaning_report() -> CleaningReport:
    return CleaningReport(
        steps=[
            CleanStepResult(
                name="drop_duplicate_rows", description="dropped 1 exact duplicate row(s)"
            ),
            CleanStepResult(
                name="handle_missing_values",
                description="filled missing values in 'amount' (2 filled with median 40.00)",
            ),
        ]
    )


def test_narrator_disabled_without_model():
    narrator = Narrator(model=None)

    assert not narrator.enabled
    assert narrator.narrate(_sample_profile(), _sample_cleaning_report()) is None


def test_build_prompt_includes_dataset_shape():
    prompt = build_prompt(_sample_profile(), _sample_cleaning_report())

    assert "Rows: 10" in prompt
    assert "Columns: 2" in prompt
    assert "Duplicate rows removed during cleaning: 1" in prompt


def test_build_prompt_describes_numeric_column():
    prompt = build_prompt(_sample_profile(), _sample_cleaning_report())

    assert "'amount'" in prompt
    assert "20.0% missing" in prompt
    assert "mean=42.5" in prompt
    assert "outliers=1" in prompt


def test_build_prompt_describes_categorical_top_values():
    prompt = build_prompt(_sample_profile(), _sample_cleaning_report())

    assert "'category'" in prompt
    assert "'widgets': 6" in prompt


def test_build_prompt_includes_cleaning_steps():
    prompt = build_prompt(_sample_profile(), _sample_cleaning_report())

    assert "drop_duplicate_rows: dropped 1 exact duplicate row(s)" in prompt
    assert "handle_missing_values" in prompt


def test_narrate_calls_local_ollama_server_and_returns_response(mock_ollama_url):
    narrator = Narrator(model="llama3.2", host=mock_ollama_url)

    result = narrator.narrate(_sample_profile(), _sample_cleaning_report())

    assert result == _MOCK_OLLAMA_REPLY


def test_narrate_raises_when_ollama_unreachable(unreachable_ollama_url):
    narrator = Narrator(model="llama3.2", host=unreachable_ollama_url)

    with pytest.raises(OllamaUnavailableError, match="couldn't reach Ollama"):
        narrator.narrate(_sample_profile(), _sample_cleaning_report())
