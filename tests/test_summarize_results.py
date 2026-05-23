import json

from benchmarks.summarize_results import row_from_summary


def test_summary_recomputes_raw_and_marks_ineligible(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps(
            {
                "ok": True,
                "total_latency_s": 1.0,
                "ttft_s": 0.2,
                "tpot_s": 0.1,
                "output_tokens": 8,
                "token_count_source": "chars_div_4_estimate",
                "has_nan_or_inf_text": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = tmp_path / "raw.summary.json"
    summary.write_text(
        json.dumps(
            {
                "run_status": "real",
                "validity_status": "valid_uncontended",
                "ok_requests": 1,
                "total_requests": 1,
                "error_requests": 0,
                "token_count_exact": False,
                "token_count_sources": ["chars_div_4_estimate"],
                "nan_or_inf_responses": 0,
                "raw_out": str(raw),
                "metadata": {"gpu_locked": True, "framework": "other", "model": "m"},
            }
        ),
        encoding="utf-8",
    )

    row = row_from_summary(summary)

    assert row["headline_eligible"] is False
    assert row["ok_requests"] == 1
    assert row["token_count_sources"] == "chars_div_4_estimate"
    assert "too_few_requests_for_claim" in row["validation_issues"]


def test_summary_reads_wrapper_gpu_before_after(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [
        {
            "ok": True,
            "total_latency_s": 1.0,
            "ttft_s": 0.2,
            "tpot_s": 0.1,
            "output_tokens": 8,
            "token_count_source": "usage.completion_tokens",
            "has_nan_or_inf_text": False,
        }
        for _ in range(10)
    ]
    raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    run_dir = tmp_path / "gpu_run"
    run_dir.mkdir()
    before = {
        "gpus": [
            {"index": 0, "memory_used_mb": 1000, "memory_free_mb": 96000},
            {"index": 1, "memory_used_mb": 2000, "memory_free_mb": 95000},
        ],
        "processes": [{"gpu_index": 0, "pid": 1}],
    }
    after = {
        "gpus": [
            {"index": 0, "memory_used_mb": 1500, "memory_free_mb": 95500},
            {"index": 1, "memory_used_mb": 2000, "memory_free_mb": 95000},
        ],
        "processes": [{"gpu_index": 0, "pid": 1}],
    }
    (run_dir / "gpu_before.json").write_text(json.dumps(before), encoding="utf-8")
    (run_dir / "gpu_after.json").write_text(json.dumps(after), encoding="utf-8")

    summary = tmp_path / "raw.summary.json"
    summary.write_text(
        json.dumps(
            {
                "run_status": "real",
                "validity_status": "valid_uncontended",
                "ok_requests": 10,
                "total_requests": 10,
                "error_requests": 0,
                "latency_p95_s": 1.0,
                "token_count_exact": True,
                "token_count_sources": ["usage.completion_tokens"],
                "nan_or_inf_responses": 0,
                "raw_out": str(raw),
                "metadata": {
                    "gpu_locked": True,
                    "gpu_ids": "0",
                    "gpu_run_dir": str(run_dir),
                    "framework": "other",
                    "model": "m",
                    "backend": "mock",
                    "dtype": "float16",
                    "extra_metadata": {"python_modules": {"torch": "test"}},
                },
            }
        ),
        encoding="utf-8",
    )

    row = row_from_summary(summary)

    assert row["gpu_before_used_mb"] == 1000
    assert row["gpu_after_used_mb"] == 1500
    assert row["gpu_used_delta_mb"] == 500
    assert row["gpu_before_process_count"] == 1
