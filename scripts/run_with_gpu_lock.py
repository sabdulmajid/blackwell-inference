#!/usr/bin/env python3
"""Run a command while holding host-local GPU locks and logging contention.

This is not a cluster scheduler. It prevents this project from launching overlapping
jobs and records whether unrelated GPU processes were present.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Allow importing sibling gpu_guard.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_guard  # noqa: E402

LOCK_DIR = Path(
    os.environ.get("BLACKWELL_INFERENCE_GPU_LOCK_DIR")
    or os.environ.get("SM120_GPU_LOCK_DIR", "/tmp/blackwell-inference-gpu-locks")
)


def now_slug() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "gpu_command"


def acquire_locks(gpu_ids: list[int], wait: bool, poll_seconds: float) -> list[Any]:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    handles = []
    for gid in sorted(gpu_ids):
        path = LOCK_DIR / f"gpu{gid}.lock"
        fh = path.open("a+")
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fh.seek(0)
                fh.truncate()
                fh.write(json.dumps({"pid": os.getpid(), "gpu": gid, "timestamp_utc": now_slug()}) + "\n")
                fh.flush()
                handles.append(fh)
                break
            except BlockingIOError:
                if not wait:
                    raise RuntimeError(f"GPU {gid} lock is already held: {path}")
                print(f"[gpu-lock] GPU {gid} lock busy; sleeping {poll_seconds}s", file=sys.stderr)
                time.sleep(poll_seconds)
    return handles


def release_locks(handles: list[Any]) -> None:
    for fh in reversed(handles):
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def annotate_child_summaries(run_dir: Path, meta: dict[str, Any]) -> None:
    for path in run_dir.rglob("*.summary.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metadata = obj.setdefault("metadata", {})
        metadata["gpu_wrapper_meta"] = meta
        metadata["contention_label"] = meta["contention_label"]
        metadata["contended"] = meta["contended"]
        metadata["gpu_status_before"] = meta["gpu_status_before"]
        metadata["gpu_status_after"] = meta["gpu_status_after"]
        if obj.get("run_status") != "dry_run":
            obj["validity_status"] = meta["contention_label"]
        write_json(path, obj)


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
    parser = argparse.ArgumentParser(description="Run command with GPU lock and nvidia-smi checks.")
    parser.add_argument("--gpus", required=True, type=parse_gpu_ids, help="Comma-separated GPU ids, e.g. 0 or 0,1")
    parser.add_argument("--min-free-gb", type=float, default=70.0)
    parser.add_argument("--allow-processes", action="store_true", help="Allow existing compute processes; run marked contended")
    parser.add_argument("--wait", action="store_true", help="Wait for project-local lock and GPU eligibility")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-wait-seconds", type=float, default=None, help="Maximum eligibility wait time before recording wait_timeout")
    parser.add_argument("--log-dir", type=Path, default=Path("results/gpu_runs"))
    parser.add_argument("--label", default="gpu_command")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    args = parser.parse_args()

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("missing command after --")

    gpu_ids = args.gpus
    run_id = f"{now_slug()}_{safe_label(args.label)}"
    run_dir = args.log_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    locks = []
    start = dt.datetime.now(dt.timezone.utc)
    rc = 99
    try:
        locks = acquire_locks(gpu_ids, wait=args.wait, poll_seconds=args.poll_seconds)

        wait_start = time.monotonic()
        while True:
            ok, reason, before = gpu_guard.eligible(gpu_ids, args.min_free_gb, args.allow_processes)
            write_json(run_dir / "gpu_before.json", before)
            if ok:
                break
            before_target_processes = gpu_guard.processes_on_gpus(before.get("processes", []), gpu_ids)
            failure_contention_label = (
                "invalid_contended" if before_target_processes else "invalid_uncertain_not_eligible"
            )
            if not args.wait:
                print(f"[gpu-lock] Not eligible: {reason}", file=sys.stderr)
                end = dt.datetime.now(dt.timezone.utc)
                write_json(run_dir / "run_meta.json", {
                    "status": "not_eligible",
                    "label": args.label,
                    "reason": reason,
                    "start_utc": start.isoformat(),
                    "end_utc": end.isoformat(),
                    "duration_seconds": (end - start).total_seconds(),
                    "gpu_ids": gpu_ids,
                    "cuda_visible_devices": None,
                    "min_free_gb": args.min_free_gb,
                    "allow_processes": args.allow_processes,
                    "command": cmd,
                    "run_dir": str(run_dir),
                    "gpu_status_before": str(run_dir / "gpu_before.json"),
                    "target_processes_before": before_target_processes,
                    "contention_label": failure_contention_label,
                    "contended": bool(before_target_processes),
                })
                return 3
            waited = time.monotonic() - wait_start
            if args.max_wait_seconds is not None and waited >= args.max_wait_seconds:
                print(f"[gpu-lock] Wait timeout after {waited:.1f}s: {reason}", file=sys.stderr)
                write_json(run_dir / "run_meta.json", {
                    "status": "wait_timeout",
                    "label": args.label,
                    "reason": reason,
                    "waited_seconds": waited,
                    "max_wait_seconds": args.max_wait_seconds,
                    "gpu_ids": gpu_ids,
                    "min_free_gb": args.min_free_gb,
                    "allow_processes": args.allow_processes,
                    "command": cmd,
                    "run_dir": str(run_dir),
                    "gpu_status_before": str(run_dir / "gpu_before.json"),
                    "target_processes_before": before_target_processes,
                    "contention_label": failure_contention_label,
                    "contended": bool(before_target_processes),
                })
                return 4
            print(f"[gpu-lock] Not eligible: {reason}; sleeping {args.poll_seconds}s", file=sys.stderr)
            time.sleep(args.poll_seconds)

        env = os.environ.copy()
        physical_gpu_ids = ",".join(str(g) for g in gpu_ids)
        before_target_processes = gpu_guard.processes_on_gpus(before.get("processes", []), gpu_ids)
        pre_contention_label = "valid_uncontended" if not before_target_processes else "invalid_contended"
        env["CUDA_VISIBLE_DEVICES"] = physical_gpu_ids
        env["BLACKWELL_INFERENCE_GPU_LOCKED"] = "1"
        env["BLACKWELL_INFERENCE_GPU_IDS"] = physical_gpu_ids
        env["BLACKWELL_INFERENCE_GPU_RUN_DIR"] = str(run_dir)
        env["BLACKWELL_INFERENCE_CUDA_VISIBLE_DEVICES_PHYSICAL"] = physical_gpu_ids
        env["BLACKWELL_INFERENCE_CONTENTION_LABEL"] = pre_contention_label
        # Legacy names are kept so older repro scripts remain runnable.
        env["SM120_LAB_GPU_LOCKED"] = "1"
        env["SM120_LAB_GPU_IDS"] = physical_gpu_ids
        env["SM120_LAB_RUN_DIR"] = str(run_dir)

        print(f"[gpu-lock] Running on GPUs {gpu_ids}: {' '.join(cmd)}", file=sys.stderr)
        with (run_dir / "stdout.log").open("w") as stdout, (run_dir / "stderr.log").open("w") as stderr:
            proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True, env=env)
            rc = proc.wait()

        after = gpu_guard.snapshot()
        write_json(run_dir / "gpu_after.json", after)
        after_target_processes = gpu_guard.processes_on_gpus(after.get("processes", []), gpu_ids)
        contention_label = (
            "valid_uncontended"
            if not before_target_processes and not after_target_processes
            else "invalid_contended"
        )

        end = dt.datetime.now(dt.timezone.utc)
        meta = {
            "status": "completed" if rc == 0 else "failed",
            "returncode": rc,
            "label": args.label,
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "duration_seconds": (end - start).total_seconds(),
            "gpu_ids": gpu_ids,
            "cuda_visible_devices": physical_gpu_ids,
            "min_free_gb": args.min_free_gb,
            "allow_processes": args.allow_processes,
            "command": cmd,
            "run_dir": str(run_dir),
            "gpu_status_before": str(run_dir / "gpu_before.json"),
            "gpu_status_after": str(run_dir / "gpu_after.json"),
            "target_processes_before": before_target_processes,
            "target_processes_after": after_target_processes,
            "contention_label": contention_label,
            "contended": contention_label != "valid_uncontended",
        }
        write_json(run_dir / "run_meta.json", meta)
        annotate_child_summaries(run_dir, meta)
        print(json.dumps(meta, indent=2))
        return rc
    finally:
        release_locks(locks)


if __name__ == "__main__":
    raise SystemExit(main())
