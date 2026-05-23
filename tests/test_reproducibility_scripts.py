import json
import subprocess
import sys

from scripts.archive_results import build_manifest, write_archive
from scripts.collect_versions import sanitize_env_value, sanitized_environment


def test_collect_versions_redacts_secret_and_paths(tmp_path):
    env = {
        "HF_TOKEN": "placeholder",
        "HF_HOME": str(tmp_path / "hf-cache"),
        "CUDA_VISIBLE_DEVICES": "0",
    }

    sanitized = sanitized_environment(env, tmp_path)

    assert sanitized["HF_TOKEN"]["value"] == "<redacted-secret>"
    assert sanitized["HF_HOME"]["value"] == "<path-redacted>"
    assert sanitized["CUDA_VISIBLE_DEVICES"]["value"] == "0"


def test_path_list_values_are_redacted(tmp_path):
    value = sanitize_env_value("LD_LIBRARY_PATH", "/a:/b:/c", tmp_path, include_path_values=False)

    assert value == {"set": True, "entries": 3, "value": "<path-list-redacted>"}


def test_archive_results_dry_manifest_and_archive(tmp_path):
    result_file = tmp_path / "results" / "x.json"
    result_file.parent.mkdir()
    result_file.write_text('{"ok": true}\n', encoding="utf-8")

    manifest = build_manifest([result_file.parent])
    assert manifest["file_count"] == 1
    assert manifest["files"][0]["path"].endswith("x.json")

    out = tmp_path / "archive.tar.gz"
    result = write_archive([result_file.parent], out, force=False)
    assert out.exists()
    assert result["file_count"] == 1
    assert (tmp_path / "archive.tar.gz.manifest.json").exists()


def test_check_reproducibility_cli_writes_json(tmp_path):
    out = tmp_path / "check.json"
    proc = subprocess.run(
        [sys.executable, "scripts/check_reproducibility.py", "--out", str(out)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert "warnings" in data
