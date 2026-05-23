#!/usr/bin/env python3
"""Check whether the repo is reproducible without mutating the environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_versions import collect_versions  # noqa: E402

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "docs/reproducibility.md",
    "docs/dependency_matrix.md",
    "docs/model_access_plan.md",
    "scripts/gpu_guard.py",
    "scripts/run_with_gpu_lock.py",
    "scripts/verify_blackwell.py",
]

BASE_PACKAGES = ["requests", "psutil", "PyYAML", "pandas", "numpy", "pytest"]
DEV_PACKAGES = ["ruff"]
OPTIONAL_PACKAGES = ["torch", "triton", "vllm", "sglang", "sgl-kernel", "flashinfer-python"]


def check(repo_root: Path) -> dict[str, Any]:
    versions = collect_versions(repo_root)
    packages = versions["packages"]
    issues = []
    warnings = []

    if sys.version_info < (3, 10):
        issues.append("Python >=3.10 is required.")

    for rel in REQUIRED_FILES:
        if not (repo_root / rel).exists():
            issues.append(f"missing required file: {rel}")

    missing_base = [pkg for pkg in BASE_PACKAGES if packages.get(pkg) is None]
    if missing_base:
        issues.append(f"missing base harness packages: {', '.join(missing_base)}")

    missing_optional = [pkg for pkg in OPTIONAL_PACKAGES if packages.get(pkg) is None]
    if missing_optional:
        warnings.append(f"missing optional framework packages: {', '.join(missing_optional)}")
    missing_dev = [pkg for pkg in DEV_PACKAGES if packages.get(pkg) is None]
    if missing_dev:
        warnings.append(f"missing developer/lint packages: {', '.join(missing_dev)}")

    if versions["python"].get("virtual_env_inside_repo") is False:
        warnings.append("current Python virtual environment is outside the repo; use .venv for reproduction")
    if not (repo_root / ".venv").exists():
        warnings.append("repo-local .venv does not exist yet")

    if versions["git"].get("this_repo") is None:
        warnings.append("this repo has no git commit yet; result metadata will record null commit")
    if versions["git"].get("external_vllm") is None:
        warnings.append("external/vllm is absent")
    if versions["git"].get("external_sglang") is None:
        warnings.append("external/sglang is absent")

    return {
        "schema_version": 1,
        "status": "ok" if not issues else "failed",
        "issues": issues,
        "warnings": warnings,
        "versions": versions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check blackwell-inference reproducibility state.")
    parser.add_argument("--out", type=Path, default=Path("results/env/reproducibility_check.json"))
    parser.add_argument("--strict", action="store_true", help="Return nonzero when warnings are present.")
    args = parser.parse_args()

    result = check(Path.cwd())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "status": result["status"], "warnings": len(result["warnings"])}, indent=2))
    if result["issues"] or (args.strict and result["warnings"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
