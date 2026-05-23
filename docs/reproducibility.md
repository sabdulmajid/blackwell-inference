# Reproducibility Guide

This guide describes a fresh, repo-local setup. It intentionally avoids hidden
global Python state, untracked model caches, and undocumented environment
variables.

## Fresh Clone Setup

```bash
git clone <REPO_URL> blackwell-inference
cd blackwell-inference

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements.txt

python scripts/collect_versions.py --out results/env/versions.json
python scripts/check_reproducibility.py --out results/env/reproducibility_check.json
pytest -q
```

The current machine has useful framework packages in another environment, but
that should be treated as observed state, not as a reproducible setup.

## Dependency Install Order

1. Base harness dependencies from `requirements.txt`.
2. Optional external source checkouts from `docs/external_repo_setup.md`.
3. Optional framework installs inside `.venv` or another named isolated env.
4. Model cache/download approval, if needed.

Do not install system packages or mutate global Python packages without approval.

## Optional vLLM Source Path

Record the checkout before editing:

```bash
bash scripts/bootstrap_external.sh --help
# Or use the sparse commands in docs/external_repo_setup.md.
git -C external/vllm status --short
git -C external/vllm rev-parse HEAD
```

Any vLLM hardware/model run must use `scripts/run_with_gpu_lock.py`.

## Optional SGLang Source Path

Use `docs/external_repo_setup.md`, then revalidate current source paths before
editing because remote `main` may have moved.

```bash
git -C external/sglang status --short
git -C external/sglang rev-parse HEAD
```

Do not install SGLang into global Python. Keep SGLang runtime repros locked and
short.

## Dry-Run Checks

```bash
python scripts/collect_versions.py --out results/env/versions.json
python scripts/check_reproducibility.py --out results/env/reproducibility_check.json

python benchmarks/serve_bench.py \
  --dry-run \
  --framework vllm \
  --model hf-internal-testing/tiny-random-gpt2 \
  --out results/benchmarks/repro_dry_run.jsonl

python benchmarks/summarize_results.py \
  --results results/benchmarks \
  --out results/benchmarks/summary.csv \
  --markdown-out results/benchmarks/summary.md
```

Dry runs validate schemas only. They are not benchmark evidence.

## One-GPU Smoke Test

Before any GPU command:

```bash
python scripts/gpu_guard.py status
python scripts/gpu_guard.py check --gpus 0 --min-free-gb 70
```

If eligible and the model is already available or explicitly approved:

```bash
python scripts/run_with_gpu_lock.py \
  --gpus 0 \
  --min-free-gb 70 \
  --wait \
  --max-wait-seconds 600 \
  --label vllm_tiny_smoke \
  -- \
  bash repros/vllm_mxfp4_sm120/run_repro.sh \
    --model <TINY_MODEL_ID_OR_LOCAL_PATH> \
    --tp 1 \
    --dtype auto
```

Do not use both GPUs until a one-GPU run is valid and uncontended.

## Target Repros

vLLM target command shape:

```bash
python scripts/run_with_gpu_lock.py --gpus 0 --min-free-gb 70 --wait \
  --label vllm_mxfp4_repro -- \
  bash repros/vllm_mxfp4_sm120/run_repro.sh \
    --model <MXFP4_MODEL_ID_OR_LOCAL_PATH> \
    --tp 1 \
    --dtype auto \
    --quantization mxfp4
```

SGLang target command shape:

```bash
python scripts/run_with_gpu_lock.py --gpus 0 --min-free-gb 70 --wait \
  --label sglang_fp8_triton -- \
  bash repros/sglang_attention_backend_sm120/run_repro.sh \
    --model <QWEN3_NEXT_FP8_MODEL_ID_OR_LOCAL_PATH> \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --linear-attn-decode-backend triton \
    --linear-attn-prefill-backend triton
```

These are command shapes, not approval to download or run large target models.

## GPU Contention

Follow `docs/gpu_contention_policy.md`:

- every GPU-consuming command goes through `scripts/run_with_gpu_lock.py`,
- use exactly one physical GPU for smoke work,
- record physical GPU IDs and wrapper metadata,
- mark contended/invalid runs honestly,
- never kill or modify other users' processes.

## Result Archiving

Preview an archive:

```bash
python scripts/archive_results.py --dry-run --out results/archives/results_preview.tar.gz
```

Create one without deleting sources:

```bash
python scripts/archive_results.py --out results/archives/results_$(date -u +%Y%m%dT%H%M%SZ).tar.gz
```

The script refuses to overwrite existing archives unless `--force` is provided.

## Schema Validation

Run:

```bash
pytest -q
python benchmarks/summarize_results.py \
  --results results/benchmarks \
  --out results/benchmarks/summary.csv \
  --markdown-out results/benchmarks/summary.md
```

Use `docs/evidence_ledger.md` to decide which artifacts support claims. Do not
cite dry-run, skipped, failed, or contended rows as benchmark results.
