#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_external.sh [--help]

Clone or update external vLLM and SGLang source trees under external/.
This script does not download models and does not run GPU workloads.
Commit SHAs are written under results/external_commits/.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 0 ]]; then
  echo "unknown arg: $1" >&2
  usage >&2
  exit 2
fi

mkdir -p external
mkdir -p results/external_commits

if [[ ! -d external/vllm/.git ]]; then
  git clone https://github.com/vllm-project/vllm.git external/vllm
else
  git -C external/vllm fetch --all --tags
fi

if [[ ! -d external/sglang/.git ]]; then
  git clone https://github.com/sgl-project/sglang.git external/sglang
else
  git -C external/sglang fetch --all --tags
fi

git -C external/vllm rev-parse HEAD | tee results/external_commits/vllm_commit.txt
git -C external/sglang rev-parse HEAD | tee results/external_commits/sglang_commit.txt
