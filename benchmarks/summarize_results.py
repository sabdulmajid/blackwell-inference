#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((q / 100.0) * (len(values) - 1)))))
    return values[idx]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def recompute_from_raw(raw_path: Path) -> dict[str, Any]:
    rows = load_jsonl(raw_path)
    ok = [r for r in rows if r.get("ok") is True]
    lat = [r["total_latency_s"] for r in ok if r.get("total_latency_s") is not None]
    ttft = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    tpot = [r["tpot_s"] for r in ok if r.get("tpot_s") is not None]
    toks = sum(r.get("output_tokens") or r.get("approx_output_tokens") or 0 for r in ok)
    return {
        "raw_total_requests": len(rows),
        "raw_ok_requests": len(ok),
        "raw_error_requests": len(rows) - len(ok),
        "raw_latency_p50_s": percentile(lat, 50),
        "raw_latency_p95_s": percentile(lat, 95),
        "raw_latency_p99_s": percentile(lat, 99),
        "raw_ttft_median_s": statistics.median(ttft) if ttft else None,
        "raw_tpot_median_s": statistics.median(tpot) if tpot else None,
        "raw_output_tokens": toks,
        "raw_nan_or_inf_responses": sum(1 for r in ok if r.get("has_nan_or_inf_text")),
        "raw_token_count_sources": sorted({r.get("token_count_source") for r in ok}),
    }


def _parse_gpu_ids(value: Any) -> list[int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return [value]
    ids = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            return None
    return ids or None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _gpu_memory_snapshot(snapshot: dict[str, Any] | None, gpu_ids: list[int] | None) -> dict[str, Any]:
    if not snapshot:
        return {}
    target_ids = set(gpu_ids or [])
    gpus = snapshot.get("gpus") or []
    selected = [
        gpu
        for gpu in gpus
        if not target_ids or gpu.get("index") in target_ids
    ]
    processes = snapshot.get("processes") or []
    selected_processes = [
        proc
        for proc in processes
        if not target_ids or proc.get("gpu_index") in target_ids
    ]
    if not selected:
        return {"gpu_process_count": len(selected_processes)}
    return {
        "gpu_memory_used_mb": sum(gpu.get("memory_used_mb") or 0 for gpu in selected),
        "gpu_memory_free_mb": sum(gpu.get("memory_free_mb") or 0 for gpu in selected),
        "gpu_process_count": len(selected_processes),
    }


def wrapper_gpu_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    run_dir_value = meta.get("gpu_run_dir")
    if not run_dir_value:
        return {}
    run_dir = Path(run_dir_value)
    gpu_ids = _parse_gpu_ids(meta.get("gpu_ids"))
    before = _gpu_memory_snapshot(_load_json(run_dir / "gpu_before.json"), gpu_ids)
    after = _gpu_memory_snapshot(_load_json(run_dir / "gpu_after.json"), gpu_ids)
    out = {
        "gpu_before_used_mb": before.get("gpu_memory_used_mb"),
        "gpu_before_free_mb": before.get("gpu_memory_free_mb"),
        "gpu_before_process_count": before.get("gpu_process_count"),
        "gpu_after_used_mb": after.get("gpu_memory_used_mb"),
        "gpu_after_free_mb": after.get("gpu_memory_free_mb"),
        "gpu_after_process_count": after.get("gpu_process_count"),
    }
    if out["gpu_before_used_mb"] is not None and out["gpu_after_used_mb"] is not None:
        out["gpu_used_delta_mb"] = out["gpu_after_used_mb"] - out["gpu_before_used_mb"]
    else:
        out["gpu_used_delta_mb"] = None
    return out


def add_validation_issues(obj: dict[str, Any], meta: dict[str, Any], issues: list[str]) -> None:
    if obj.get("run_status") == "dry_run":
        return
    if not meta.get("extra_metadata"):
        issues.append("missing_env_metadata")
    if not meta.get("backend") or meta.get("backend") == "unknown":
        issues.append("ambiguous_backend")
    if not meta.get("dtype") or meta.get("dtype") == "unknown":
        issues.append("ambiguous_dtype")
    total_requests = obj.get("total_requests") or 0
    if total_requests < 10:
        issues.append("too_few_requests_for_claim")
    if total_requests < 20 and (obj.get("latency_p95_s") is not None or obj.get("latency_p99_s") is not None):
        issues.append("p95_p99_low_sample_count")


def headline_eligible(obj: dict[str, Any], meta: dict[str, Any], raw: dict[str, Any], issues: list[str]) -> bool:
    return (
        obj.get("run_status") == "real"
        and obj.get("validity_status") == "valid_uncontended"
        and meta.get("gpu_locked") is True
        and obj.get("ok_requests") == obj.get("total_requests")
        and raw.get("raw_ok_requests") == raw.get("raw_total_requests")
        and obj.get("token_count_exact") is True
        and raw.get("raw_nan_or_inf_responses") == 0
        and not issues
    )


def row_from_summary(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    meta = obj.get("metadata", {})
    issues = []
    raw_path = Path(obj.get("raw_out", ""))
    raw = {}
    if raw_path.exists():
        try:
            raw = recompute_from_raw(raw_path)
        except Exception as e:
            issues.append(f"raw_parse_failed:{e}")
    else:
        issues.append("raw_missing")
    if obj.get("run_status") != "dry_run" and raw and raw.get("raw_total_requests") != obj.get("total_requests"):
        issues.append("summary_raw_total_mismatch")
    if obj.get("run_status") != "dry_run" and raw and raw.get("raw_ok_requests") != obj.get("ok_requests"):
        issues.append("summary_raw_ok_mismatch")
    add_validation_issues(obj, meta, issues)
    row = {
        "path": str(path),
        "raw_path": str(raw_path) if raw_path else None,
        "timestamp_utc": meta.get("timestamp_utc"),
        "framework": meta.get("framework"),
        "framework_commit": meta.get("framework_commit"),
        "model": meta.get("model"),
        "backend": meta.get("backend"),
        "dtype": meta.get("dtype"),
        "quantization": meta.get("quantization"),
        "tp": meta.get("tensor_parallel"),
        "input_words": meta.get("input_words"),
        "max_tokens": meta.get("max_tokens"),
        "concurrency": meta.get("concurrency"),
        "warmup": meta.get("warmup"),
        "run_status": obj.get("run_status"),
        "validity_status": obj.get("validity_status"),
        "headline_eligible": headline_eligible(obj, meta, raw, issues),
        "validation_issues": ";".join(issues),
        "ok_requests": obj.get("ok_requests"),
        "total_requests": obj.get("total_requests"),
        "error_requests": obj.get("error_requests"),
        "agg_output_toks_s": obj.get("aggregate_output_tokens_per_s"),
        "requests_s": obj.get("requests_per_s"),
        "latency_p50_s": obj.get("latency_p50_s"),
        "latency_p95_s": obj.get("latency_p95_s"),
        "latency_p99_s": obj.get("latency_p99_s"),
        "ttft_median_s": obj.get("ttft_median_s"),
        "ttft_p95_s": obj.get("ttft_p95_s"),
        "tpot_median_s": obj.get("tpot_median_s"),
        "output_tokens": obj.get("output_tokens"),
        "token_count_exact": obj.get("token_count_exact"),
        "token_count_sources": ",".join(obj.get("token_count_sources") or []),
        "nan_or_inf_responses": obj.get("nan_or_inf_responses"),
        "gpu_locked": meta.get("gpu_locked"),
        "gpu_ids": meta.get("gpu_ids"),
        "cuda_visible_devices": meta.get("cuda_visible_devices"),
        "contention_label": meta.get("contention_label"),
        "command": " ".join(meta.get("command") or []),
    }
    row.update(wrapper_gpu_metadata(meta))
    return row


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Benchmark Summary", ""]
    if not rows:
        lines.append("No benchmark summaries found.")
    else:
        lines.append("| framework | model | backend | status | validity | headline eligible | latency p50 | TTFT median | output tok/s |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                "| {framework} | {model} | {backend} | {run_status} | {validity_status} | {headline_eligible} | {latency_p50_s} | {ttft_median_s} | {agg_output_toks_s} |".format(
                    **{k: row.get(k) for k in row}
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize benchmark JSONL files.")
    p.add_argument("--results", type=Path, default=Path("results/benchmarks"))
    p.add_argument("--out", type=Path, default=Path("results/benchmarks/summary.csv"))
    p.add_argument("--markdown-out", type=Path, default=Path("results/benchmarks/summary.md"))
    args = p.parse_args()

    rows = []
    for path in args.results.rglob("*.summary.json"):
        try:
            rows.append(row_from_summary(path))
        except Exception:
            rows.append({"path": str(path), "validation_issues": "summary_parse_failed", "headline_eligible": False})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.out.open("w", newline="", encoding="utf-8") as f:
            fieldnames = sorted({key for row in rows for key in row})
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    else:
        args.out.write_text("", encoding="utf-8")
    write_markdown(rows, args.markdown_out)
    print(f"wrote {len(rows)} rows to {args.out} and {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
