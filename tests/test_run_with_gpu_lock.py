import json
import sys

from scripts import run_with_gpu_lock


def test_not_eligible_meta_records_validity(monkeypatch, tmp_path):
    snap = {
        "ok": True,
        "process_query_ok": True,
        "gpus": [{"index": 0, "memory_free_gb": 60.0}],
        "processes": [{"gpu_index": 0, "pid": 123, "process_name": "python", "used_memory_mb": 1024}],
    }

    monkeypatch.setattr(run_with_gpu_lock, "acquire_locks", lambda *args, **kwargs: [])
    monkeypatch.setattr(run_with_gpu_lock, "release_locks", lambda handles: None)
    monkeypatch.setattr(
        run_with_gpu_lock.gpu_guard,
        "eligible",
        lambda gpu_ids, min_free_gb, allow_processes: (False, "target process present", snap),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_with_gpu_lock.py",
            "--gpus",
            "0",
            "--min-free-gb",
            "70",
            "--log-dir",
            str(tmp_path),
            "--label",
            "unit",
            "--",
            "true",
        ],
    )

    assert run_with_gpu_lock.main() == 3

    meta_path = next(tmp_path.glob("*_unit/run_meta.json"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert meta["status"] == "not_eligible"
    assert meta["contention_label"] == "invalid_contended"
    assert meta["contended"] is True
    assert meta["target_processes_before"] == snap["processes"]
    assert meta["cuda_visible_devices"] is None
    assert meta["run_dir"] == str(meta_path.parent)
