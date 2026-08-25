"""Shared pytest fixtures."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

MOCK_OLLAMA_REPLY = "This dataset looks mostly clean, with one column worth a second look."


class _MockOllamaHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # drain the request body; not asserted here

        body = json.dumps({"response": MOCK_OLLAMA_REPLY}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:  # noqa: A002
        pass  # keep test output quiet


@pytest.fixture
def mock_ollama_url():
    """A throwaway local server mimicking Ollama's /api/generate
    response shape, so narration runs through its real HTTP code path
    without needing an actual Ollama install."""
    server = HTTPServer(("127.0.0.1", 0), _MockOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture
def unreachable_ollama_url():
    """A URL that reliably refuses connections: bind a server to grab
    a free port, then close it without ever serving."""
    server = HTTPServer(("127.0.0.1", 0), _MockOllamaHandler)
    port = server.server_port
    server.server_close()
    return f"http://127.0.0.1:{port}"
