#!/usr/bin/env python3
"""Archive result artifacts without deleting or overwriting them."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import tarfile
from pathlib import Path
from typing import Any


def iter_files(paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            files.add(path)
            continue
        files.update(p for p in path.rglob("*") if p.is_file())
    return sorted(files)


def build_manifest(paths: list[Path]) -> dict[str, Any]:
    files = iter_files(paths)
    return {
        "schema_version": 1,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "file_count": len(files),
        "files": [{"path": str(path), "size_bytes": path.stat().st_size} for path in files],
    }


def write_archive(paths: list[Path], out: Path, force: bool) -> dict[str, Any]:
    if out.exists() and not force:
        raise FileExistsError(f"archive already exists: {out}")
    manifest = build_manifest(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for item in manifest["files"]:
            tar.add(item["path"])
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"archive": str(out), "manifest": str(manifest_path), **manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive results without deleting source artifacts.")
    parser.add_argument("--path", action="append", type=Path, default=None, help="Path to include. Repeatable.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = args.path or [Path("results"), Path("docs/evidence_ledger.md"), Path("results/README.md")]
    out = args.out or Path("results/archives") / f"results_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.tar.gz"
    manifest = build_manifest(paths)
    if args.dry_run:
        print(json.dumps({"archive": str(out), "dry_run": True, **manifest}, indent=2))
        return 0
    result = write_archive(paths, out, args.force)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
