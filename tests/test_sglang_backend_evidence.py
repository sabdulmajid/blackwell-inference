import json
import subprocess
import sys


def test_extract_sglang_backend_evidence(tmp_path):
    grep_file = tmp_path / "backend_grep.txt"
    out = tmp_path / "backend_summary.json"
    status = tmp_path / "status.txt"
    grep_file.write_text(
        "server.log:10:Linear attention kernel backend: decode=flashinfer, prefill=triton\n"
        "server.log:11:GDN kernel dispatcher: decode=flashinfer, extend=triton, verify=triton\n"
        "server.log:12:triton.runtime.errors.OutOfResources: out of resource: "
        "shared memory, Required: 114688, Hardware limit: 101376\n",
        encoding="utf-8",
    )
    status.write_text("server exited early\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/extract_sglang_backend_evidence.py",
            "--grep-file",
            str(grep_file),
            "--out",
            str(out),
            "--status-file",
            str(status),
            "--model",
            "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
            "--attention-backend",
            "triton",
            "--fp8-gemm-backend",
            "triton",
            "--linear-attn-decode-backend",
            "flashinfer",
            "--linear-attn-prefill-backend",
            "triton",
            "--tp",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["selected_linear_decode_backend"] == "flashinfer"
    assert data["selected_linear_prefill_backend"] == "triton"
    assert data["shared_memory_hits"][0]["requested_shared_memory_bytes"] == 114688
    assert data["shared_memory_hits"][0]["hardware_shared_memory_limit_bytes"] == 101376
    assert data["shared_memory_hits"][0]["exceeds_limit"] is True
    assert "triton_out_of_resources_shared_memory" in data["failure_classes"]
    assert data["run_config"]["fp8_gemm_backend"] == "triton"
