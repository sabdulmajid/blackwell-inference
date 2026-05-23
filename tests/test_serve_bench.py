import json
import subprocess
import sys

import pytest

from benchmarks import serve_bench


class FakeResponse:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        yield from self._lines


def test_chat_once_ttft_waits_for_content(monkeypatch):
    times = iter([10.0, 11.0, 12.0])
    monkeypatch.setattr(serve_bench.time, "perf_counter", lambda: next(times))
    lines = [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        'data: {"choices":[{"delta":{"content":"hello world"}}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr(serve_bench.requests, "post", lambda *a, **k: FakeResponse(lines))

    row = serve_bench.chat_once("http://server", "model", "prompt", 4, 0.0, 5.0)

    assert row["ok"] is True
    assert row["ttft_s"] == pytest.approx(1.0)
    assert row["token_count_source"] == "chars_div_4_estimate"


def test_chat_once_uses_usage_completion_tokens(monkeypatch):
    times = iter([1.0, 1.5, 2.0])
    monkeypatch.setattr(serve_bench.time, "perf_counter", lambda: next(times))
    lines = [
        'data: {"choices":[{"delta":{"content":"hello"}}]}',
        'data: {"choices":[{"delta":{}}],"usage":{"completion_tokens":7}}',
        "data: [DONE]",
    ]
    monkeypatch.setattr(serve_bench.requests, "post", lambda *a, **k: FakeResponse(lines))

    row = serve_bench.chat_once("http://server", "model", "prompt", 4, 0.0, 5.0)

    assert row["output_tokens"] == 7
    assert row["token_count_source"] == "usage.completion_tokens"


def test_chat_once_requests_stream_usage_by_default(monkeypatch):
    captured = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"hello"}}]}',
                'data: {"choices":[{"delta":{}}],"usage":{"completion_tokens":1}}',
                "data: [DONE]",
            ]
        )

    times = iter([1.0, 1.1, 1.2])
    monkeypatch.setattr(serve_bench.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(serve_bench.requests, "post", fake_post)

    row = serve_bench.chat_once("http://server", "model", "prompt", 4, 0.0, 5.0)

    assert row["ok"] is True
    assert captured["stream_options"] == {"include_usage": True}


def test_chat_once_done_only_is_not_ok(monkeypatch):
    times = iter([1.0, 2.0])
    monkeypatch.setattr(serve_bench.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(serve_bench.requests, "post", lambda *a, **k: FakeResponse(["data: [DONE]"]))

    row = serve_bench.chat_once("http://server", "model", "prompt", 4, 0.0, 5.0)

    assert row["ok"] is False
    assert row["output_tokens"] == 0


def test_nan_inf_detection_is_token_aware():
    assert serve_bench.has_nan_or_inf_text("value is NaN")
    assert serve_bench.has_nan_or_inf_text("value is inf")
    assert not serve_bench.has_nan_or_inf_text("useful information")


def test_dry_run_writes_non_metric_schema(tmp_path):
    raw = tmp_path / "dry.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "benchmarks/serve_bench.py",
            "--framework",
            "other",
            "--model",
            "dry-model",
            "--dry-run",
            "--out",
            str(raw),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    row = json.loads(raw.read_text().splitlines()[0])
    summary = json.loads(raw.with_suffix(".summary.json").read_text())
    assert row["run_status"] == "dry_run"
    assert row["metrics_valid"] is False
    assert summary["validity_status"] == "dry_run_no_server_contact"
    assert summary["error_requests"] == 0
    assert summary["token_count_exact"] is False
    assert summary["token_count_sources"] == []
    assert summary["nan_or_inf_responses"] == 0


def test_cli_rejects_invalid_requests(tmp_path):
    raw = tmp_path / "bad.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "benchmarks/serve_bench.py",
            "--framework",
            "other",
            "--model",
            "dry-model",
            "--requests",
            "0",
            "--out",
            str(raw),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "--requests must be >= 1" in proc.stderr


def test_cli_rejects_unlocked_real_requests(tmp_path):
    raw = tmp_path / "real.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "benchmarks/serve_bench.py",
            "--framework",
            "other",
            "--model",
            "model",
            "--out",
            str(raw),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "refusing to send benchmark requests" in proc.stderr
