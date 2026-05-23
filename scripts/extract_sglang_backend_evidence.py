#!/usr/bin/env python3
"""Extract structured SGLang backend and shared-memory evidence from repro logs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


BACKEND_PATTERNS = [
    re.compile(
        r"Linear attention kernel backend:\s*decode=(?P<decode>[^,\s]+),\s*prefill=(?P<prefill>[^\s]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"GDN kernel dispatcher:\s*decode=(?P<decode>[^,\s]+),\s*extend=(?P<extend>[^,\s]+),\s*verify=(?P<verify>[^\s]+)",
        re.IGNORECASE,
    ),
    re.compile(r"Using (?P<backend>[^\n]*attention backend[^\n]*)", re.IGNORECASE),
    re.compile(r"attention[_ -]backend[:= ]+(?P<backend>[A-Za-z0-9_./-]+)", re.IGNORECASE),
    re.compile(r"fp8[_ -]gemm[_ -]backend[:= ]+(?P<backend>[A-Za-z0-9_./-]+)", re.IGNORECASE),
]

SHARED_MEMORY_PATTERNS = [
    re.compile(
        r"Required(?:\s+shared\s+memory)?[:= ]+(?P<required>\d+)\s*(?:bytes|B)?.*?"
        r"(?:Hardware limit|limit)[:= ]+(?P<limit>\d+)\s*(?:bytes|B)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<required>\d+)\s*(?:bytes|B)[^\n]*(?:required|requested)[^\n]*"
        r"(?P<limit>\d+)\s*(?:bytes|B)[^\n]*(?:limit|available)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:required|requested)[^\n]*(?P<required>\d+)\s*(?:bytes|B)[^\n]*"
        r"(?:limit|available)[^\n]*(?P<limit>\d+)\s*(?:bytes|B)",
        re.IGNORECASE,
    ),
]

KEY_TERMS = (
    "outofresources",
    "shared memory",
    "hardware limit",
    "attention",
    "flashinfer",
    "triton",
    "fp8",
    "deepgemm",
    "backend",
    "qwen3",
    "gdn",
    "linear",
    "nan",
    "error",
    "exception",
)


def _maybe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def extract(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    evidence_lines: list[dict[str, Any]] = []
    backend_hits: list[dict[str, Any]] = []
    shared_memory_hits: list[dict[str, Any]] = []
    failure_classes: set[str] = set()

    for line_number, line in enumerate(lines, start=1):
        lower = line.lower()
        terms = [term for term in KEY_TERMS if term in lower]
        backend_match: dict[str, Any] | None = None
        shared_match: dict[str, Any] | None = None

        if "outofresources" in lower:
            failure_classes.add("triton_out_of_resources")
        if "outofresources" in lower and "shared" in lower:
            failure_classes.add("triton_out_of_resources_shared_memory")

        for pattern in BACKEND_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            groups = {key: value for key, value in match.groupdict().items() if value}
            backend_match = {"line_number": line_number, "line": line, **groups}
            backend_hits.append(backend_match)
            break

        for pattern in SHARED_MEMORY_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            required = _maybe_int(match.groupdict().get("required"))
            limit = _maybe_int(match.groupdict().get("limit"))
            shared_match = {
                "line_number": line_number,
                "line": line,
                "requested_shared_memory_bytes": required,
                "hardware_shared_memory_limit_bytes": limit,
                "exceeds_limit": (
                    required is not None and limit is not None and required > limit
                ),
            }
            shared_memory_hits.append(shared_match)
            failure_classes.add("triton_out_of_resources_shared_memory")
            break

        if terms or backend_match or shared_match:
            evidence_lines.append(
                {
                    "line_number": line_number,
                    "terms": terms,
                    "backend": backend_match,
                    "shared_memory": shared_match,
                    "line": line,
                }
            )

    selected_attention_backend = None
    selected_linear_decode_backend = None
    selected_linear_prefill_backend = None
    selected_fp8_gemm_backend = None
    for hit in backend_hits:
        line = hit["line"].lower()
        if "decode" in hit:
            selected_linear_decode_backend = hit["decode"]
        if "prefill" in hit:
            selected_linear_prefill_backend = hit["prefill"]
        if "attention backend" in line and "backend" in hit:
            selected_attention_backend = hit["backend"].strip()
        if "fp8" in line and "backend" in hit:
            selected_fp8_gemm_backend = hit["backend"].strip()

    return {
        "schema_version": 1,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": str(path),
        "selected_attention_backend": selected_attention_backend,
        "selected_linear_decode_backend": selected_linear_decode_backend,
        "selected_linear_prefill_backend": selected_linear_prefill_backend,
        "selected_fp8_gemm_backend": selected_fp8_gemm_backend,
        "backend_hits": backend_hits,
        "shared_memory_hits": shared_memory_hits,
        "failure_classes": sorted(failure_classes),
        "evidence_line_count": len(evidence_lines),
        "evidence_lines": evidence_lines[:300],
        "truncated": len(evidence_lines) > 300,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grep-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--fp8-gemm-backend", default=None)
    parser.add_argument("--linear-attn-backend", default=None)
    parser.add_argument("--linear-attn-decode-backend", default=None)
    parser.add_argument("--linear-attn-prefill-backend", default=None)
    parser.add_argument("--tp", default=None)
    args = parser.parse_args()

    result = extract(args.grep_file)
    result["run_status"] = (
        args.status_file.read_text(encoding="utf-8").strip()
        if args.status_file and args.status_file.exists()
        else None
    )
    result["run_config"] = {
        "model": args.model,
        "attention_backend": args.attention_backend,
        "fp8_gemm_backend": args.fp8_gemm_backend,
        "linear_attn_backend": args.linear_attn_backend,
        "linear_attn_decode_backend": args.linear_attn_decode_backend,
        "linear_attn_prefill_backend": args.linear_attn_prefill_backend,
        "tp": args.tp,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(args.out),
                "failure_classes": result["failure_classes"],
                "shared_memory_hits": len(result["shared_memory_hits"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
