import os
import subprocess


REPRO_SCRIPTS = [
    "repros/vllm_mxfp4_sm120/run_repro.sh",
    "repros/sglang_attention_backend_sm120/run_repro.sh",
]


def test_repro_scripts_parse_and_have_help():
    for script in REPRO_SCRIPTS:
        assert subprocess.run(["bash", "-n", script], check=False).returncode == 0
        help_proc = subprocess.run(["bash", script, "--help"], text=True, capture_output=True, check=False)
        assert help_proc.returncode == 0
        assert "run_with_gpu_lock.py" in help_proc.stdout


def test_repro_scripts_refuse_unlocked_launch(tmp_path):
    for script in REPRO_SCRIPTS:
        env = os.environ.copy()
        env["BLACKWELL_INFERENCE_GPU_RUN_DIR"] = str(tmp_path / script.replace("/", "_"))
        env.pop("BLACKWELL_INFERENCE_GPU_LOCKED", None)
        env.pop("SM120_LAB_GPU_LOCKED", None)
        proc = subprocess.run(
            ["bash", script, "--model", "dummy/model"],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert proc.returncode == 4
        assert "Refusing to launch" in proc.stderr


def test_repro_scripts_dry_run(tmp_path):
    for script in REPRO_SCRIPTS:
        env = os.environ.copy()
        env["BLACKWELL_INFERENCE_GPU_RUN_DIR"] = str(tmp_path / script.replace("/", "_"))
        proc = subprocess.run(
            ["bash", script, "--model", "dummy/model", "--dry-run"],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert proc.returncode == 0
        assert "dry_run" in proc.stdout
