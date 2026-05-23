import subprocess
import sys


def test_torch_cuda_smoke_help():
    proc = subprocess.run(
        [sys.executable, "scripts/torch_cuda_smoke.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "tiny PyTorch CUDA" in proc.stdout
