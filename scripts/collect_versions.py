#!/usr/bin/env python3
"""Collect sanitized reproducibility metadata without initializing CUDA."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKAGE_NAMES = [
    "requests",
    "psutil",
    "PyYAML",
    "pandas",
    "numpy",
    "pytest",
    "ruff",
    "torch",
    "triton",
    "vllm",
    "sglang",
    "sgl-kernel",
    "flashinfer-python",
    "transformers",
    "huggingface-hub",
    "accelerate",
    "safetensors",
]

ENV_EXACT_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "TRITON_CACHE_DIR",
    "XDG_CACHE_HOME",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "BLACKWELL_INFERENCE_GPU_LOCKED",
    "BLACKWELL_INFERENCE_GPU_IDS",
    "BLACKWELL_INFERENCE_GPU_RUN_DIR",
}
ENV_PREFIXES = ("HF_", "HUGGINGFACE_", "VLLM_", "SGLANG_", "FLASHINFER_", "BLACKWELL_INFERENCE_")
SECRET_MARKERS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "PASS", "CREDENTIAL", "COOKIE")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(path: Path) -> str | None:
    if not path.exists():
        return None
    proc = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_remote(path: Path) -> str | None:
    if not path.exists():
        return None
    proc = subprocess.run(
        ["git", "-C", str(path), "remote", "get-url", "origin"], text=True, capture_output=True
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def path_summary(value: str, repo_root: Path) -> dict[str, Any]:
    path = Path(value).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    try:
        inside_repo = resolved.is_relative_to(repo_root.resolve())
    except (AttributeError, OSError):
        inside_repo = str(resolved).startswith(str(repo_root.resolve()))
    return {
        "set": True,
        "value": "<path-redacted>",
        "basename": path.name,
        "inside_repo": inside_repo,
    }


def sanitize_env_value(key: str, value: str, repo_root: Path, include_path_values: bool) -> Any:
    if any(marker in key.upper() for marker in SECRET_MARKERS):
        return {"set": True, "value": "<redacted-secret>"}
    if not value:
        return {"set": False, "value": None}
    if key in {"PATH", "LD_LIBRARY_PATH", "PYTHONPATH"}:
        parts = value.split(":")
        return {"set": True, "entries": len(parts), "value": "<path-list-redacted>"}
    looks_like_path = value.startswith("/") or value.startswith("~")
    if looks_like_path and not include_path_values:
        return path_summary(value, repo_root)
    return {"set": True, "value": value}


def sanitized_environment(
    env: dict[str, str], repo_root: Path, include_path_values: bool = False
) -> dict[str, Any]:
    selected = {}
    for key in sorted(env):
        if key in ENV_EXACT_KEYS or key.startswith(ENV_PREFIXES):
            selected[key] = sanitize_env_value(key, env[key], repo_root, include_path_values)
    return selected


def collect_versions(repo_root: Path, include_path_values: bool = False) -> dict[str, Any]:
    venv = os.environ.get("VIRTUAL_ENV")
    repo_root = repo_root.resolve()
    local_venv = None
    if venv:
        try:
            local_venv = Path(venv).resolve().is_relative_to(repo_root)
        except (AttributeError, OSError):
            local_venv = str(Path(venv).resolve()).startswith(str(repo_root))
    return {
        "schema_version": 1,
        "project": "blackwell-inference",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": path_summary(sys.executable, repo_root),
            "virtual_env": path_summary(venv, repo_root) if venv else {"set": False, "value": None},
            "virtual_env_inside_repo": local_venv,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {name: package_version(name) for name in PACKAGE_NAMES},
        "git": {
            "this_repo": git_commit(repo_root),
            "external_vllm": git_commit(repo_root / "external" / "vllm"),
            "external_sglang": git_commit(repo_root / "external" / "sglang"),
            "external_vllm_remote": git_remote(repo_root / "external" / "vllm"),
            "external_sglang_remote": git_remote(repo_root / "external" / "sglang"),
        },
        "environment": sanitized_environment(os.environ, repo_root, include_path_values),
        "notes": [
            "This script does not import torch or initialize CUDA.",
            "Path-like environment values are redacted by default.",
            "Secret-like environment variable names are always redacted.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect sanitized dependency/version metadata.")
    parser.add_argument("--out", type=Path, default=Path("results/env/versions.json"))
    parser.add_argument(
        "--include-path-values",
        action="store_true",
        help="Include path-like env values. Do not use for committed artifacts.",
    )
    args = parser.parse_args()

    data = collect_versions(Path.cwd(), include_path_values=args.include_path_values)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "packages": data["packages"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
