"""Exercises Narrator against the real anthropic SDK, talking to a
local mock server instead of api.anthropic.com. This is the one code
path the unit tests in test_narrate.py can't reach with a fake client
object: real HTTP request formation, real response deserialization
into the SDK's own Message type, and our _extract_text() handling of
that real object rather than a stand-in. No network access or API key
required -- everything happens on localhost.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

anthropic = pytest.importorskip("anthropic")

from glance.cleaning import CleaningReport, CleanStepResult  # noqa: E402
from glance.narrate import Narrator  # noqa: E402
from glance.profiling import ColumnProfile, DatasetProfile  # noqa: E402

_FAKE_REPLY = "This dataset has 10 rows, one missing region, and a clear outlier in amount."


class _MockMessagesHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # drain the request body; content isn't asserted here

        body = json.dumps(
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5-20251001",
                "content": [{"type": "text", "text": _FAKE_REPLY}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 42, "output_tokens": 12},
            }
        ).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:  # noqa: A002
        pass  # keep test output quiet


@pytest.fixture
def mock_anthropic_url():
    server = HTTPServer(("127.0.0.1", 0), _MockMessagesHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def _sample_profile() -> DatasetProfile:
    return DatasetProfile(
        row_count=10,
        column_count=1,
        duplicate_row_count=0,
        columns=[
            ColumnProfile(
                name="amount",
                dtype="float64",
                missing_count=0,
                missing_pct=0.0,
                unique_count=8,
                mean=42.5,
                median=40.0,
                std=5.1,
                min=10.0,
                max=90.0,
                outlier_count=1,
            )
        ],
    )


def _sample_cleaning_report() -> CleaningReport:
    return CleaningReport(steps=[CleanStepResult(name="drop_duplicate_rows", description="none")])


def test_narrate_through_real_sdk_against_local_server(mock_anthropic_url):
    narrator = Narrator(api_key=None)
    narrator._client = anthropic.Anthropic(api_key="local-test-key", base_url=mock_anthropic_url)

    result = narrator.narrate(_sample_profile(), _sample_cleaning_report())

    assert result == _FAKE_REPLY
