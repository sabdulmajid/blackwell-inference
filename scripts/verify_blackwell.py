#!/usr/bin/env python3
"""Collect reproducible environment metadata for blackwell-inference runs."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(cmd: list[str]) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except FileNotFoundError:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": f"not found: {cmd[0]}"}
    except subprocess.TimeoutExpired as e:
        return {"cmd": cmd, "returncode": None, "stdout": e.stdout or "", "stderr": "timeout"}


def module_version(name: str) -> str | None:
    package_names = {
        "flashinfer": "flashinfer-python",
        "flashinfer_python": "flashinfer-python",
    }
    dist_name = package_names.get(name, name)
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        pass
    if importlib.util.find_spec(name) is None:
        return None
    return "installed_version_unknown"


def torch_info(probe_cuda: bool) -> dict[str, Any]:
    version = module_version("torch")
    if version is None:
        return {"installed": False, "cuda_probe": False}
    if not probe_cuda:
        return {
            "installed": True,
            "torch_version": version,
            "cuda_probe": False,
            "cuda_probe_note": "Skipped by default to avoid CUDA initialization outside the GPU lock wrapper.",
        }
    if importlib.util.find_spec("torch") is None:
        return {"installed": True, "torch_version": version, "cuda_probe": False}
    try:
        import torch
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "total_memory_bytes": props.total_memory,
                    "multi_processor_count": props.multi_processor_count,
                    "shared_memory_per_block": getattr(props, "shared_memory_per_block", None),
                    "shared_memory_per_multiprocessor": getattr(props, "shared_memory_per_multiprocessor", None),
                    "major": getattr(props, "major", None),
                    "minor": getattr(props, "minor", None),
                })
        return {
            "installed": True,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_probe": True,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpus": gpus,
        }
    except Exception as e:
        return {"installed": True, "error": str(e)}


def git_commit(path: Path) -> str | None:
    p = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True)
    return p.stdout.strip() if p.returncode == 0 else None


def gpu_guard_snapshot() -> dict[str, Any]:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import gpu_guard  # type: ignore

        return gpu_guard.snapshot()
    except Exception as e:
        return {"ok": False, "error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Blackwell environment metadata.")
    parser.add_argument("--out", type=Path, default=Path("results/env/verify_blackwell.json"))
    parser.add_argument("--vllm-path", type=Path, default=Path("external/vllm"))
    parser.add_argument("--sglang-path", type=Path, default=Path("external/sglang"))
    parser.add_argument(
        "--probe-cuda",
        action="store_true",
        help="Import torch and query torch.cuda device properties. Use under run_with_gpu_lock.py.",
    )
    args = parser.parse_args()

    locked = (
        os.environ.get("BLACKWELL_INFERENCE_GPU_LOCKED") == "1"
        or os.environ.get("SM120_LAB_GPU_LOCKED") == "1"
    )
    if args.probe_cuda and not locked:
        parser.error("--probe-cuda may initialize CUDA and must be run under scripts/run_with_gpu_lock.py")

    data: dict[str, Any] = {
        "project": "blackwell-inference",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "cwd": os.getcwd(),
        },
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "BLACKWELL_INFERENCE_GPU_LOCKED": os.environ.get("BLACKWELL_INFERENCE_GPU_LOCKED"),
            "BLACKWELL_INFERENCE_GPU_IDS": os.environ.get("BLACKWELL_INFERENCE_GPU_IDS"),
            "BLACKWELL_INFERENCE_GPU_RUN_DIR": os.environ.get("BLACKWELL_INFERENCE_GPU_RUN_DIR"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
        "commands": {
            "nvidia_smi": run(["nvidia-smi"]),
            "nvidia_smi_query": run([
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,pcie.link.gen.current,pcie.link.width.current,power.draw,power.limit,temperature.gpu",
                "--format=csv",
            ]),
            "nvidia_smi_compute_apps": run([
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv",
            ]),
            "nvidia_smi_topo": run(["nvidia-smi", "topo", "-m"]),
            "nvcc": run(["nvcc", "--version"]),
        },
        "gpu_guard": gpu_guard_snapshot(),
        "python_modules": {
            "torch": module_version("torch"),
            "triton": module_version("triton"),
            "vllm": module_version("vllm"),
            "sglang": module_version("sglang"),
            "flashinfer": module_version("flashinfer"),
            "flashinfer_python": module_version("flashinfer_python"),
        },
        "torch": torch_info(args.probe_cuda),
        "git": {
            "this_repo": git_commit(Path.cwd()),
            "vllm": git_commit(args.vllm_path) if args.vllm_path.exists() else None,
            "sglang": git_commit(args.sglang_path) if args.sglang_path.exists() else None,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "torch_cuda_available": data["torch"].get("cuda_available")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
