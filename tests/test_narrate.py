from types import SimpleNamespace

from glance.cleaning import CleaningReport, CleanStepResult
from glance.narrate import Narrator, _extract_text, build_prompt
from glance.profiling import ColumnProfile, DatasetProfile


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


def test_narrator_disabled_without_api_key():
    narrator = Narrator(api_key=None)

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


def test_extract_text_from_text_block():
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")])
    assert _extract_text(response) == "hello"


def test_extract_text_skips_non_text_blocks():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", text=None),
            SimpleNamespace(type="text", text="the actual summary"),
        ]
    )
    assert _extract_text(response) == "the actual summary"


def test_extract_text_no_text_block_returns_empty_string():
    response = SimpleNamespace(content=[])
    assert _extract_text(response) == ""


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.last_call_kwargs = None

    def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._response


class _FakeClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


def test_narrate_calls_client_with_expected_model_and_extracts_result():
    narrator = Narrator(api_key=None)  # avoid constructing a real client
    fake_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="This dataset has 10 rows.")]
    )
    narrator._client = _FakeClient(fake_response)  # inject the fake

    result = narrator.narrate(_sample_profile(), _sample_cleaning_report())

    assert result == "This dataset has 10 rows."
    assert narrator._client.messages.last_call_kwargs["model"] == "claude-haiku-4-5-20251001"
    assert narrator._client.messages.last_call_kwargs["max_tokens"] == 500
