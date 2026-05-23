from scripts import gpu_guard


def test_eligible_rejects_target_process(monkeypatch):
    monkeypatch.setattr(
        gpu_guard,
        "snapshot",
        lambda: {
            "ok": True,
            "process_query_ok": True,
            "gpus": [{"index": 0, "memory_free_gb": 80.0}],
            "processes": [{"gpu_index": 0, "pid": 123, "process_name": "python"}],
        },
    )

    ok, reason, _ = gpu_guard.eligible([0], 70.0, allow_processes=False)

    assert not ok
    assert "active compute processes" in reason


def test_eligible_ignores_non_target_process(monkeypatch):
    monkeypatch.setattr(
        gpu_guard,
        "snapshot",
        lambda: {
            "ok": True,
            "process_query_ok": True,
            "gpus": [{"index": 0, "memory_free_gb": 80.0}, {"index": 1, "memory_free_gb": 10.0}],
            "processes": [{"gpu_index": 1, "pid": 123, "process_name": "python"}],
        },
    )

    ok, reason, _ = gpu_guard.eligible([0], 70.0, allow_processes=False)

    assert ok
    assert reason == "eligible"


def test_eligible_rejects_unknown_free_memory(monkeypatch):
    monkeypatch.setattr(
        gpu_guard,
        "snapshot",
        lambda: {
            "ok": True,
            "process_query_ok": True,
            "gpus": [{"index": 0, "memory_free_gb": None}],
            "processes": [],
        },
    )

    ok, reason, _ = gpu_guard.eligible([0], 70.0, allow_processes=False)

    assert not ok
    assert "free memory is unknown" in reason


def test_eligible_rejects_failed_process_query(monkeypatch):
    monkeypatch.setattr(
        gpu_guard,
        "snapshot",
        lambda: {
            "ok": True,
            "process_query_ok": False,
            "process_query_error": "query failed",
            "gpus": [{"index": 0, "memory_free_gb": 80.0}],
            "processes": [],
        },
    )

    ok, reason, _ = gpu_guard.eligible([0], 70.0, allow_processes=False)

    assert not ok
    assert "process query failed" in reason
