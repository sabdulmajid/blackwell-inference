# Benchmark Matrix Ramp-Up

Status: dry-run validated; real one-GPU benchmark blocked by active external GPU
processes at the required `--min-free-gb 70` threshold.

## Current Availability

Artifacts:

- `results/gpu_status/benchmark_matrix_pre_status.json`
- `results/gpu_status/benchmark_matrix_gpu0_check.json`
- `results/gpu_status/benchmark_matrix_gpu1_check.json`
- `results/gpu_status/hardware_sentinel_20260520_current_status.json`
- `results/gpu_status/hardware_sentinel_20260520_gpu0_check.json`
- `results/gpu_status/hardware_sentinel_20260520_gpu1_check.json`

Current blocker:

- GPU 0 had about `63.25` GiB free, below the required `70` GiB threshold.
- GPU 1 had about `67.07` GiB free, below the required `70` GiB threshold.
- Both GPUs had active `python` PID `2999453`.

No vLLM or SGLang serving benchmark was launched under these conditions.

## Stage A: One-GPU Tiny/Small Smoke

Dry-run schema artifacts:

- vLLM tiny dry run:
  `results/benchmarks/20260520T103000Z_stage_a_vllm_tiny_dry_run.jsonl`
- vLLM tiny summary:
  `results/benchmarks/20260520T103000Z_stage_a_vllm_tiny_dry_run.summary.json`
- SGLang tiny dry run:
  `results/benchmarks/20260520T103001Z_stage_a_sglang_tiny_dry_run.jsonl`
- SGLang tiny summary:
  `results/benchmarks/20260520T103001Z_stage_a_sglang_tiny_dry_run.summary.json`

Planned first real vLLM smoke, once a single GPU is eligible:

```bash
python scripts/gpu_guard.py status --out results/gpu_status/vllm_tiny_smoke_pre_status.json

python scripts/run_with_gpu_lock.py \
  --gpus 1 \
  --min-free-gb 70 \
  --wait \
  --max-wait-seconds 600 \
  --label vllm_tiny_smoke \
  -- \
  bash repros/vllm_mxfp4_sm120/run_repro.sh \
    --model hf-internal-testing/tiny-random-gpt2 \
    --tp 1 \
    --dtype auto
```

SGLang Stage A is blocked until SGLang is installed or checked out in an isolated
environment.

## Stage B: One-GPU Meaningful Benchmark

Dry-run schema artifacts:

- vLLM meaningful dry run:
  `results/benchmarks/20260520T102402Z_stage_b_vllm_meaningful_dry_run.jsonl`
- SGLang meaningful dry run:
  `results/benchmarks/20260520T102403Z_stage_b_sglang_meaningful_dry_run.jsonl`

Candidate model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, if download/storage is
approved and the GPU is eligible. Do not compare this to target FP4/FP8 repros.

## Stage C: Target Repro Benchmarks

vLLM target:

- Goal: SM120 FP4/MXFP4/NVFP4 backend-selection evidence.
- Current artifact: `results/repros/vllm_mxfp4_sm120/static_backend_selection_probe.json`.
- Runtime blocker: no eligible GPU window and no approved small MXFP4/NVFP4
  checkpoint that exercises the target codepath.

SGLang target:

- Goal: FP8 attention/backend/shared-memory behavior for Qwen3-Next-style hybrid
  GDN models.
- Current artifacts:
  `results/repros/sglang_attention_backend_sm120/sglang_triton_dry_run_20260520T101900Z/command.txt`
  and
  `results/repros/sglang_attention_backend_sm120/sglang_flashinfer_decode_dry_run_20260520T101901Z/command.txt`.
- Runtime blocker: SGLang is not installed and no small local GDN fixture is
  available.

## Stage D: Two-GPU Benchmark

Not eligible yet. Run TP=2 only after:

1. Stage A has at least one `valid_uncontended` one-GPU real serving benchmark.
2. The same framework/model/backend has a clean TP=1 result.
3. Both GPUs pass `gpu_guard.py check --gpus 0,1 --min-free-gb 70`.
4. Topology and before/after GPU status are recorded in the wrapper run
   directory.

## Current Summary

The summary files are:

- `results/benchmarks/summary.csv`
- `results/benchmarks/summary.md`

All current benchmark rows are dry runs and have `headline_eligible=False`.
They validate schema and artifact plumbing only; they do not support any
performance claim.
