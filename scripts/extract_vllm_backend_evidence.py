#!/usr/bin/env python3
"""Extract structured vLLM backend-selection evidence from repro logs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


BACKEND_PATTERNS = [
    re.compile(r"Using '(?P<backend>[^']+)' Mxfp4 MoE backend", re.IGNORECASE),
    re.compile(r"Using '(?P<backend>[^']+)' NvFp4 MoE backend", re.IGNORECASE),
    re.compile(r"Using (?P<backend>FlashInfer MXFP4 [^\\n]+ backend[^\\n]*)", re.IGNORECASE),
    re.compile(r"Using (?P<backend>Marlin backend)", re.IGNORECASE),
    re.compile(r"Using (?P<backend>Triton backend)", re.IGNORECASE),
    re.compile(r"Falling back to (?P<backend>Marlin FP4 MoE kernel)", re.IGNORECASE),
]

KEY_TERMS = (
    "mxfp4",
    "nvfp4",
    "fp4",
    "marlin",
    "flashinfer",
    "cutlass",
    "triton",
    "deepgemm",
    "backend",
    "quant",
    "fallback",
    "error",
    "exception",
)


def extract(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    evidence = []
    backend_hits = []
    for idx, line in enumerate(lines, start=1):
        lower = line.lower()
        terms = [term for term in KEY_TERMS if term in lower]
        matched_backend = None
        for pattern in BACKEND_PATTERNS:
            match = pattern.search(line)
            if match:
                matched_backend = match.group("backend").strip()
                backend_hits.append({"line_number": idx, "backend": matched_backend, "line": line})
                break
        if terms or matched_backend:
            evidence.append(
                {
                    "line_number": idx,
                    "terms": terms,
                    "backend": matched_backend,
                    "line": line,
                }
            )
    selected = backend_hits[-1]["backend"] if backend_hits else None
    return {
        "schema_version": 1,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": str(path),
        "selected_backend": selected,
        "backend_hits": backend_hits,
        "evidence_line_count": len(evidence),
        "evidence_lines": evidence[:200],
        "truncated": len(evidence) > 200,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grep-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--quantization", default=None)
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
        "dtype": args.dtype,
        "quantization": args.quantization,
        "tp": args.tp,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "selected_backend": result["selected_backend"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
