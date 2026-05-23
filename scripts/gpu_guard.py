#!/usr/bin/env python3
"""GPU status and lightweight contention checker for blackwell-inference.

This script is intentionally conservative. It does not kill processes and does not
assume that file locks prevent external users from starting GPU jobs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

LOCK_DIR = Path(
    os.environ.get("BLACKWELL_INFERENCE_GPU_LOCK_DIR")
    or os.environ.get("SM120_GPU_LOCK_DIR", "/tmp/blackwell-inference-gpu-locks")
)


@dataclass
class GpuInfo:
    index: int
    name: str | None = None
    uuid: str | None = None
    memory_total_mb: int | None = None
    memory_used_mb: int | None = None
    memory_free_mb: int | None = None
    utilization_gpu: int | None = None
    temperature_gpu: int | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None

    @property
    def memory_free_gb(self) -> float | None:
        if self.memory_free_mb is None:
            return None
        return self.memory_free_mb / 1024.0


@dataclass
class GpuProcess:
    gpu_index: int | None
    pid: int
    process_name: str | None
    used_memory_mb: int | None


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        raise RuntimeError(f"Command not found: {cmd[0]}") from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{e.output}") from None


def parse_int(s: str) -> int | None:
    s = s.strip()
    if not s or s.upper() == "N/A":
        return None
    digits = "".join(ch for ch in s if ch.isdigit() or ch == "-")
    return int(digits) if digits else None


def parse_float(s: str) -> float | None:
    s = s.strip()
    if not s or s.upper() == "N/A":
        return None
    allowed = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
    return float(allowed) if allowed else None


def query_gpus() -> list[GpuInfo]:
    fields = [
        "index",
        "name",
        "uuid",
        "memory.total",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "temperature.gpu",
        "power.draw",
        "power.limit",
    ]
    out = run([
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ])
    rows = csv.reader(out.strip().splitlines())
    gpus: list[GpuInfo] = []
    for row in rows:
        if not row:
            continue
        row = [x.strip() for x in row]
        gpus.append(
            GpuInfo(
                index=int(row[0]),
                name=row[1],
                uuid=row[2],
                memory_total_mb=parse_int(row[3]),
                memory_used_mb=parse_int(row[4]),
                memory_free_mb=parse_int(row[5]),
                utilization_gpu=parse_int(row[6]),
                temperature_gpu=parse_int(row[7]),
                power_draw_w=parse_float(row[8]),
                power_limit_w=parse_float(row[9]),
            )
        )
    return gpus


def query_processes() -> list[GpuProcess]:
    fields = ["gpu_uuid", "pid", "process_name", "used_memory"]
    out = run([
        "nvidia-smi",
        f"--query-compute-apps={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ])
    gpu_uuid_to_index = {g.uuid: g.index for g in query_gpus()}
    procs: list[GpuProcess] = []
    for row in csv.reader(out.strip().splitlines()):
        if not row or len(row) < 2:
            continue
        row = [x.strip() for x in row]
        if row[0].upper() == "N/A" or not row[1].isdigit():
            continue
        procs.append(
            GpuProcess(
                gpu_index=gpu_uuid_to_index.get(row[0]),
                pid=int(row[1]),
                process_name=row[2] if len(row) > 2 else None,
                used_memory_mb=parse_int(row[3]) if len(row) > 3 else None,
            )
        )
    return procs


def snapshot() -> dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        gpus = query_gpus()
        topo = subprocess.run(["nvidia-smi", "topo", "-m"], text=True, capture_output=True)
        topo_text = topo.stdout if topo.returncode == 0 else topo.stderr
        process_query_ok = True
        process_query_error = None
        try:
            procs = query_processes()
        except RuntimeError as e:
            procs = []
            process_query_ok = False
            process_query_error = str(e)
        return {
            "timestamp_utc": now,
            "ok": True,
            "lock_dir": str(LOCK_DIR),
            "gpus": [asdict(g) | {"memory_free_gb": g.memory_free_gb} for g in gpus],
            "processes": [asdict(p) for p in procs],
            "process_query_ok": process_query_ok,
            "process_query_error": process_query_error,
            "topology": topo_text,
        }
    except Exception as e:
        return {"timestamp_utc": now, "ok": False, "error": str(e)}


def processes_on_gpus(processes: list[dict[str, Any]] | list[GpuProcess], gpu_ids: list[int]) -> list[dict[str, Any]]:
    targets = set(gpu_ids)
    found = []
    for proc in processes:
        item = asdict(proc) if isinstance(proc, GpuProcess) else proc
        if item.get("gpu_index") in targets:
            found.append(item)
    return found


def eligible(gpu_ids: list[int], min_free_gb: float, allow_processes: bool) -> tuple[bool, str, dict[str, Any]]:
    snap = snapshot()
    if not snap.get("ok"):
        return False, f"GPU status failed: {snap.get('error')}", snap
    if not snap.get("process_query_ok", False):
        return False, f"GPU process query failed: {snap.get('process_query_error')}", snap
    by_id = {g["index"]: g for g in snap["gpus"]}
    for gid in gpu_ids:
        if gid not in by_id:
            return False, f"GPU {gid} not found", snap
        free = by_id[gid].get("memory_free_gb")
        if free is None:
            return False, f"GPU {gid} free memory is unknown; need {min_free_gb:.2f}GB", snap
        if free < min_free_gb:
            return False, f"GPU {gid} has only {free:.2f}GB free; need {min_free_gb:.2f}GB", snap
    target_procs = processes_on_gpus(snap["processes"], gpu_ids)
    if target_procs and not allow_processes:
        return False, f"Target GPUs have active compute processes: {target_procs}", snap
    return True, "eligible", snap


def parse_gpu_ids(value: str) -> list[int]:
    try:
        gpu_ids = [int(x) for x in value.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid GPU id list: {value}") from e
    if not gpu_ids:
        raise argparse.ArgumentTypeError("at least one GPU id is required")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError(f"duplicate GPU id in list: {value}")
    return gpu_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Check GPU status and contention.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    status_p = sub.add_parser("status", help="Print full GPU status JSON")
    status_p.add_argument("--out", type=Path, default=None)

    check_p = sub.add_parser("check", help="Check whether GPUs are eligible for a run")
    check_p.add_argument("--gpus", required=True, type=parse_gpu_ids, help="Comma-separated GPU ids, e.g. 0 or 0,1")
    check_p.add_argument("--min-free-gb", type=float, default=70.0)
    check_p.add_argument("--allow-processes", action="store_true")
    check_p.add_argument("--out", type=Path, default=None)

    args = parser.parse_args()

    if args.cmd == "status":
        snap = snapshot()
        text = json.dumps(snap, indent=2)
        print(text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")
        return 0 if snap.get("ok") else 2

    if args.cmd == "check":
        ok, reason, snap = eligible(args.gpus, args.min_free_gb, args.allow_processes)
        result = {"eligible": ok, "reason": reason, "snapshot": snap}
        text = json.dumps(result, indent=2)
        print(text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")
        return 0 if ok else 3

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
