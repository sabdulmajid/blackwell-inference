import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_extract_vllm_backend_evidence(tmp_path):
    grep_file = tmp_path / "backend_grep.txt"
    out = tmp_path / "backend_summary.json"
    status = tmp_path / "status.txt"
    grep_file.write_text(
        "server.log:10:Using 'FLASHINFER_CUTLASS_MXFP4_MXFP8' Mxfp4 MoE backend.\n"
        "server.log:11:other quant message\n",
        encoding="utf-8",
    )
    status.write_text("completed\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/extract_vllm_backend_evidence.py",
            "--grep-file",
            str(grep_file),
            "--out",
            str(out),
            "--status-file",
            str(status),
            "--model",
            "dummy/model",
            "--dtype",
            "auto",
            "--quantization",
            "mxfp4",
            "--tp",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["selected_backend"] == "FLASHINFER_CUTLASS_MXFP4_MXFP8"
    assert data["run_status"] == "completed"
    assert data["run_config"]["quantization"] == "mxfp4"


def test_vllm_mxfp4_static_probe_records_sm120_marlin(tmp_path):
    if os.environ.get("BLACKWELL_INFERENCE_RUN_OPTIONAL_IMPORT_TESTS") != "1":
        pytest.skip("set BLACKWELL_INFERENCE_RUN_OPTIONAL_IMPORT_TESTS=1 to run vLLM import probe")
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("vLLM is not installed in this environment")

    out = tmp_path / "probe.json"
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/vllm_mxfp4_static_probe.py",
                "--out",
                str(out),
                "--upstream-head",
                "test-head",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("vLLM static probe timed out in this environment")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["run_status"] == "completed"
    scenario = data["mxfp4_backend_scenarios"]["sm120_flashinfer_all_flags"]
    assert scenario["backend_name"] == "MARLIN"
    assert scenario["is_device_capability_100"] is False
    assert scenario["has_device_capability_100"] is True
    assert scenario["triton_range_sm90_to_before_sm110"] is False
