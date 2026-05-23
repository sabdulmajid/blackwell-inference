#!/usr/bin/env python3
"""OpenAI-compatible streaming benchmark client.

Launch vLLM/SGLang separately, then run this client against the server.
The benchmark measures client-observed TTFT and total latency. Token counts are
estimated from streamed chunks unless the server returns usage metadata.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import statistics
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

BAD_NUMERIC_RE = re.compile(r"(?<![A-Za-z0-9_])(?:nan|inf|infinity)(?![A-Za-z0-9_])", re.IGNORECASE)


def make_prompt(input_words: int, seed: int) -> str:
    base = (
        "You are benchmarking an LLM serving system. "
        "Respond with concise factual text. "
        "The following filler keeps the prompt length stable: "
    )
    filler_words = [f"w{(i + seed) % 1000}" for i in range(max(0, input_words))]
    return base + " ".join(filler_words)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((q / 100.0) * (len(values) - 1)))))
    return values[idx]


def git_commit(path: Path = Path(".")) -> str | None:
    p = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True)
    return p.stdout.strip() if p.returncode == 0 else None


def gpu_env_metadata() -> dict[str, Any]:
    gpu_ids = os.environ.get("BLACKWELL_INFERENCE_GPU_IDS") or os.environ.get("SM120_LAB_GPU_IDS")
    run_dir = os.environ.get("BLACKWELL_INFERENCE_GPU_RUN_DIR") or os.environ.get("SM120_LAB_RUN_DIR")
    contention_label = os.environ.get("BLACKWELL_INFERENCE_CONTENTION_LABEL")
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_locked": (
            os.environ.get("BLACKWELL_INFERENCE_GPU_LOCKED") == "1"
            or os.environ.get("SM120_LAB_GPU_LOCKED") == "1"
        ),
        "gpu_ids": gpu_ids,
        "gpu_run_dir": run_dir,
        "wrapper_contention_label": contention_label,
    }


def validity_status(contention_label: str) -> str:
    if contention_label.startswith("valid_"):
        return contention_label
    if contention_label.startswith("invalid_contended"):
        return "contended"
    if contention_label.startswith("invalid_"):
        return contention_label
    return "unverified"


def build_metadata(args: argparse.Namespace) -> dict[str, Any]:
    gpu_meta = gpu_env_metadata()
    contention_label = (
        args.gpu_contention_label
        or gpu_meta.get("wrapper_contention_label")
        or "unknown_until_wrapper_finishes"
    )
    if not args.dry_run and not gpu_meta["gpu_locked"] and args.gpu_contention_label is None:
        contention_label = "invalid_uncertain_unlocked"
    meta = {
        "schema_version": 1,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "command": sys.argv,
        "framework": args.framework,
        "framework_commit": args.framework_commit,
        "model": args.model,
        "backend": args.backend,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "tensor_parallel": args.tp,
        "base_url": args.base_url,
        "input_words": args.input_words,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "warmup": args.warmup,
        "temperature": args.temperature,
        "include_usage": args.include_usage,
        "dry_run": args.dry_run,
        "contention_label": contention_label,
        **gpu_meta,
    }
    if args.metadata and args.metadata.exists():
        meta["extra_metadata"] = json.loads(args.metadata.read_text())
    return meta


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def dry_run_summary(meta: dict[str, Any], raw_out: Path) -> dict[str, Any]:
    return {
        "run_status": "dry_run",
        "validity_status": "dry_run_no_server_contact",
        "ok_requests": 0,
        "total_requests": 0,
        "error_requests": 0,
        "elapsed_s": 0.0,
        "aggregate_output_tokens_per_s": None,
        "requests_per_s": None,
        "latency_p50_s": None,
        "latency_p95_s": None,
        "latency_p99_s": None,
        "ttft_median_s": None,
        "ttft_p95_s": None,
        "tpot_median_s": None,
        "output_tokens": 0,
        "token_count_exact": False,
        "token_count_sources": [],
        "nan_or_inf_responses": 0,
        "metadata": meta,
        "raw_out": str(raw_out),
    }


def has_nan_or_inf_text(text: str) -> bool:
    return BAD_NUMERIC_RE.search(text) is not None


def chat_once(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    include_usage: bool = True,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    headers = {"Content-Type": "application/json"}
    t0 = time.perf_counter()
    first = None
    chunks = 0
    chars = 0
    text_parts = []
    status_code = None
    error = None
    usage_completion_tokens = None
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as r:
            status_code = r.status_code
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("data: "):
                    raw = raw[len("data: ") :]
                if raw.strip() == "[DONE]":
                    break
                chunks += 1
                try:
                    obj = json.loads(raw)
                    usage = obj.get("usage")
                    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                        usage_completion_tokens = int(usage["completion_tokens"])
                    delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        if first is None:
                            first = time.perf_counter()
                        text_parts.append(delta)
                        chars += len(delta)
                except Exception:
                    if first is None:
                        first = time.perf_counter()
                    chars += len(raw)
    except Exception as e:
        error = str(e)
    t1 = time.perf_counter()
    generated_text = "".join(text_parts)
    if usage_completion_tokens is not None:
        output_tokens = usage_completion_tokens
        token_count_source = "usage.completion_tokens"
    else:
        # Approximate without usage. This supports smoke comparisons, not headline claims.
        output_tokens = max(1, round(chars / 4)) if chars else 0
        token_count_source = "chars_div_4_estimate" if chars else "none"
    ttft = (first - t0) if first is not None else None
    total = t1 - t0
    tpot = ((total - ttft) / max(1, output_tokens - 1)) if ttft is not None and output_tokens > 1 else None
    ok = (
        error is None
        and status_code is not None
        and 200 <= status_code < 300
        and output_tokens > 0
        and first is not None
    )
    return {
        "request_id": str(uuid.uuid4()),
        "status_code": status_code,
        "ok": ok,
        "error": error,
        "ttft_s": ttft,
        "total_latency_s": total,
        "tpot_s": tpot,
        "chunks": chunks,
        "approx_output_tokens": output_tokens,
        "output_tokens": output_tokens,
        "token_count_source": token_count_source,
        "chars": chars,
        "output_tokens_per_s": (output_tokens / total) if total > 0 else None,
        "output_preview": generated_text[:200],
        "has_nan_or_inf_text": has_nan_or_inf_text(generated_text),
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.requests < 1 and not args.dry_run:
        raise SystemExit("--requests must be >= 1 unless --dry-run is used")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.input_words < 0:
        raise SystemExit("--input-words must be >= 0")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be >= 1")
    if args.out.exists() and args.out.stat().st_size > 0 and not args.append:
        raise SystemExit(f"output exists and is non-empty; use --append or choose a new --out: {args.out}")
    if not args.dry_run and not args.allow_unlocked_client:
        gpu_meta = gpu_env_metadata()
        if not gpu_meta["gpu_locked"]:
            raise SystemExit(
                "refusing to send benchmark requests without scripts/run_with_gpu_lock.py; "
                "use --allow-unlocked-client only for non-GPU/local test servers"
            )


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark OpenAI-compatible LLM server.")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--framework", choices=["vllm", "sglang", "other"], required=True)
    p.add_argument("--backend", default="unknown")
    p.add_argument("--dtype", default="unknown")
    p.add_argument("--quantization", default="unknown")
    p.add_argument("--framework-commit", default=None)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--input-words", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--requests", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument(
        "--include-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request streamed usage metadata for exact completion token counts when supported.",
    )
    p.add_argument("--out", type=Path, default=Path("results/benchmarks/raw.jsonl"))
    p.add_argument("--summary-out", type=Path, default=None)
    p.add_argument("--append", action="store_true", help="Append to existing JSONL instead of refusing to overwrite it")
    p.add_argument("--metadata", type=Path, default=None, help="Optional JSON metadata file to merge")
    p.add_argument("--gpu-contention-label", default=None)
    p.add_argument(
        "--allow-unlocked-client",
        action="store_true",
        help="Allow requests without GPU-lock env; only for non-GPU/local test servers.",
    )
    p.add_argument("--dry-run", action="store_true", help="Write schema/example outputs without contacting a server")
    args = p.parse_args()
    validate_args(args)

    meta = build_metadata(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.append:
        args.out.write_text("", encoding="utf-8")
    summary_path = args.summary_out or args.out.with_suffix(".summary.json")

    if args.dry_run:
        row = {
            "request_id": "dry-run",
            "run_status": "dry_run",
            "metrics_valid": False,
            "ok": None,
            "error": None,
            "ttft_s": None,
            "total_latency_s": None,
            "tpot_s": None,
            "chunks": 0,
            "approx_output_tokens": None,
            "output_tokens": None,
            "token_count_source": "none",
            "chars": 0,
            "output_tokens_per_s": None,
            "output_preview": "",
            "has_nan_or_inf_text": None,
            "metadata": meta,
        }
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        summary = dry_run_summary(meta, args.out)
        write_json(summary_path, summary)
        print(json.dumps(summary, indent=2))
        return 0

    # Warmup sequentially.
    for i in range(args.warmup):
        chat_once(
            args.base_url,
            args.model,
            make_prompt(args.input_words, i),
            args.max_tokens,
            args.temperature,
            args.timeout,
            args.include_usage,
        )

    results = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = []
        for i in range(args.requests):
            futs.append(
                ex.submit(
                    chat_once,
                    args.base_url,
                    args.model,
                    make_prompt(args.input_words, i),
                    args.max_tokens,
                    args.temperature,
                    args.timeout,
                    args.include_usage,
                )
            )
        for fut in as_completed(futs):
            row = fut.result()
            row["run_status"] = "real"
            row["metrics_valid"] = bool(row["ok"])
            row["metadata"] = meta
            results.append(row)
            with args.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    elapsed = time.perf_counter() - t_start

    ok = [r for r in results if r["ok"]]
    lat = [r["total_latency_s"] for r in ok]
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    tpot = [r["tpot_s"] for r in ok if r["tpot_s"] is not None]
    toks = sum(r["output_tokens"] for r in ok)
    exact_tokens = all(r.get("token_count_source") == "usage.completion_tokens" for r in ok) if ok else False
    run_status = "real" if len(ok) == len(results) else "failed"
    summary = {
        "run_status": run_status,
        "validity_status": validity_status(meta["contention_label"]),
        "ok_requests": len(ok),
        "total_requests": len(results),
        "error_requests": len(results) - len(ok),
        "elapsed_s": elapsed,
        "aggregate_output_tokens_per_s": toks / elapsed if elapsed > 0 else None,
        "requests_per_s": len(ok) / elapsed if elapsed > 0 else None,
        "latency_p50_s": percentile(lat, 50),
        "latency_p95_s": percentile(lat, 95),
        "latency_p99_s": percentile(lat, 99),
        "ttft_median_s": statistics.median(ttft) if ttft else None,
        "ttft_p95_s": percentile(ttft, 95),
        "tpot_median_s": statistics.median(tpot) if tpot else None,
        "output_tokens": toks,
        "token_count_exact": exact_tokens,
        "token_count_sources": sorted({r.get("token_count_source") for r in ok}),
        "nan_or_inf_responses": sum(1 for r in ok if r.get("has_nan_or_inf_text")),
        "metadata": meta,
        "raw_out": str(args.out),
    }
    print(json.dumps(summary, indent=2))
    write_json(summary_path, summary)
    return 0 if len(ok) == len(results) else 4


if __name__ == "__main__":
    raise SystemExit(main())
