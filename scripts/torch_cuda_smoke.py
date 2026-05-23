#!/usr/bin/env python3
"""Tiny real CUDA smoke test for blackwell-inference.

This script intentionally does not load a model. It verifies that a locked process
can initialize PyTorch CUDA, allocate a very small tensor, run one matmul, and
write machine-readable evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny PyTorch CUDA allocation/matmul smoke test.")
    parser.add_argument("--out", type=Path, required=True, help="Path for JSON result.")
    parser.add_argument("--size", type=int, default=256, help="Square matrix size. Default: 256.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    start = dt.datetime.now(dt.timezone.utc)
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_status": "started",
        "timestamp_utc": start.isoformat(),
        "hostname": platform.node(),
        "python": sys.version,
        "command": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_locked": os.environ.get("BLACKWELL_INFERENCE_GPU_LOCKED") == "1",
        "physical_gpu_ids": os.environ.get("BLACKWELL_INFERENCE_GPU_IDS"),
        "gpu_run_dir": os.environ.get("BLACKWELL_INFERENCE_GPU_RUN_DIR"),
        "matrix_size": args.size,
        "dtype": args.dtype,
    }

    try:
        import torch

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]
        if not torch.cuda.is_available():
            result.update({"run_status": "failed", "error": "torch.cuda.is_available() is false"})
            write_json(args.out, result)
            return 2

        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        result.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "logical_device": 0,
                "device_name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": props.total_memory,
                "memory_allocated_before_bytes": torch.cuda.memory_allocated(0),
                "memory_reserved_before_bytes": torch.cuda.memory_reserved(0),
            }
        )

        torch.manual_seed(1234)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        a = torch.randn((args.size, args.size), device="cuda", dtype=dtype)
        b = torch.randn((args.size, args.size), device="cuda", dtype=dtype)
        c = a @ b
        checksum = float(c.float().sum().detach().cpu().item())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        result.update(
            {
                "run_status": "completed",
                "elapsed_seconds": elapsed,
                "checksum": checksum,
                "memory_allocated_after_bytes": torch.cuda.memory_allocated(0),
                "memory_reserved_after_bytes": torch.cuda.memory_reserved(0),
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(0),
            }
        )
        write_json(args.out, result)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        result.update({"run_status": "failed", "error": repr(e)})
        write_json(args.out, result)
        print(json.dumps(result, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
