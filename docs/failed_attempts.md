# Failed Attempts

## 2026-05-23: meaningful-goal CUDA smoke blocked by another user's process

### Goal

Run the smallest real GPU-affecting CUDA smoke test and produce actual GPU
evidence without violating the contention policy.

### Planned command

```bash
python scripts/run_with_gpu_lock.py \
  --gpus 0 \
  --min-free-gb 70 \
  --wait \
  --poll-seconds 30 \
  --max-wait-seconds 900 \
  --label meaningful_goal_torch_cuda_smoke \
  -- \
  bash -lc 'python scripts/torch_cuda_smoke.py --out "$BLACKWELL_INFERENCE_GPU_RUN_DIR/torch_cuda_smoke.json" --size 256 --dtype float16'
```

### Outcome

The CUDA smoke did not start. The wrapper waited for 906.5 seconds and recorded
a structured `wait_timeout` because physical GPU 0 had an active Python process
for the entire wait window.

The active process was inspected with `ps` and appeared to belong to another
user:

```text
PID 507867 python /3d-data/y2863claude/recipe-research/oracle/run_oracle_sweep.py ...
```

No process was killed, suspended, reniced, or otherwise disturbed.

Evidence:

- Plan: `results/plans/meaningful_goal_gpu_smoke_plan.md`
- Start status: `results/gpu_status/meaningful_goal_start_status.json`
- GPU 0 check: `results/gpu_status/meaningful_goal_gpu0_check.json`
- GPU 1 check: `results/gpu_status/meaningful_goal_gpu1_check.json`
- Wait-timeout artifact:
  `results/gpu_runs/20260523T153219Z_meaningful_goal_torch_cuda_smoke/run_meta.json`
- Final status: `results/gpu_status/meaningful_goal_final_status.json`

### Classification

`blocked: gpu contention`

### Next action

Retry the same command only after:

```bash
python scripts/gpu_guard.py check --gpus 0 --min-free-gb 70
```

returns eligible. If GPU 1 becomes eligible first, update the plan to target
GPU 1 and record the selection rationale.

## 2026-05-23: impact pass stayed source-only because GPUs were active

### Goal

Make a meaningful next step without violating GPU contention policy.

### Outcome

No GPU-heavy workload was launched. A fresh GPU status check showed active
Python work on both GPUs, so this pass used sparse source inspection instead of
a runtime repro.

Evidence:

- Status: `results/gpu_status/impact_resume_status.json`
- Source audit: `docs/current_upstream_source_audit.md`
- vLLM checkout: `external/vllm` at
  `5bb8d2767a2829b56e58c68fa8f380e9e4e2bd3e`
- SGLang checkout: `external/sglang` at
  `a5a64a311a39b153d1e4d3d6bcb67e77cdc9aeae`

### Classification

`blocked: gpu contention`, with `completed: current-upstream source audit`

### Next action

Retry a locked one-GPU runtime smoke only after:

```bash
python scripts/gpu_guard.py check --gpus 0 --min-free-gb 70
```

returns eligible, or use GPU 1 only if it is the eligible least-contended GPU
and the selection rationale is recorded.

## 2026-05-20: one-GPU PyTorch CUDA smoke blocked by active GPU process

### Goal

Run the smallest real GPU-affecting smoke test through `scripts/run_with_gpu_lock.py`.

### Planned command

```bash
python scripts/run_with_gpu_lock.py \
  --gpus 1 \
  --min-free-gb 70 \
  --wait \
  --poll-seconds 15 \
  --max-wait-seconds 600 \
  --label torch_cuda_smoke_gpu1 \
  -- \
  bash -lc 'python scripts/torch_cuda_smoke.py --out "$BLACKWELL_INFERENCE_GPU_RUN_DIR/torch_cuda_smoke.json" --size 256 --dtype float16'
```

### Selection rationale

GPU 0 was not safe because `results/gpu_status/pre_real_smoke_gpu0_check.json` showed active process PID `2805853`.
GPU 1 was selected as the least-contended GPU because `results/gpu_status/pre_real_smoke_gpu1_check.json` showed slightly less memory in use and more free VRAM than GPU 0.

### Outcome

The CUDA smoke did not start. The wrapper waited for GPU 1 to become eligible, but active process PID `2805853` remained present on GPU 1.

Evidence:

- Plan: `results/plans/one_gpu_smoke_plan.md`
- Initial status: `results/gpu_status/pre_real_smoke_status.json`
- GPU 0 check: `results/gpu_status/pre_real_smoke_gpu0_check.json`
- GPU 1 check: `results/gpu_status/pre_real_smoke_gpu1_check.json`
- External bounded wait artifact: `results/gpu_runs/20260520T093325Z_torch_cuda_smoke_gpu1/gpu_before.json`
- Structured wait-timeout artifact: `results/gpu_runs/20260520T094432Z_torch_cuda_smoke_gpu1_waitcheck/run_meta.json`
- Post-attempt status: `results/gpu_status/post_real_smoke_waitcheck_status.json`

### Classification

`blocked: gpu contention`

### Harness change made

Added `scripts/torch_cuda_smoke.py` as the minimal real CUDA smoke path and added `--max-wait-seconds` to `scripts/run_with_gpu_lock.py` so future blocked attempts write a structured `wait_timeout` `run_meta.json`.

### Next action

Retry the same one-GPU smoke only after:

```bash
python scripts/gpu_guard.py check --gpus 1 --min-free-gb 70
```

returns eligible, or after GPU 0 becomes eligible and the plan is updated to target GPU 0.

## 2026-05-20: vLLM real MXFP4 repro deferred by active GPU process

### Goal

Run the smallest safe vLLM SM120 MXFP4 backend-selection repro.

### Outcome

No vLLM GPU command was launched. Guard checks showed both GPUs had active
compute process PID `2805853`, so the pass stayed on static source/probe
evidence only.

Evidence:

- Status: `results/gpu_status/vllm_deep_dive_status_latest.json`
- GPU 0 check: `results/gpu_status/vllm_deep_dive_gpu0_check.json`
- GPU 1 check: `results/gpu_status/vllm_deep_dive_gpu1_check.json`
- Static selector probe:
  `results/repros/vllm_mxfp4_sm120/static_backend_selection_probe.json`

### Classification

`blocked: gpu contention`

### Next action

Retry the one-GPU vLLM repro only after exactly one target GPU is eligible:

```bash
python scripts/gpu_guard.py check --gpus 1 --min-free-gb 70
```

## 2026-05-20: SGLang runtime repro deferred by missing local runtime

### Goal

Run the smallest safe SGLang RTX PRO 6000 Blackwell FP8 attention/backend
reproducer.

### Outcome

No SGLang server or model command was launched. `external/sglang` is absent and
the current Python environment does not have `sglang` or `sgl-kernel`
installed. The pass stayed on upstream source mapping, a locked local SM120 CUDA
property probe, and dry-run repro command artifacts.

Evidence:

- Technical note: `docs/sglang_sm120_attention_backend.md`
- Locked device probe:
  `results/gpu_runs/20260520T100750Z_sglang_device_probe/verify_blackwell_probe_cuda.json`
- Final GPU status:
  `results/gpu_status/sglang_deep_dive_final_status.json`
- Triton dry-run command:
  `results/repros/sglang_attention_backend_sm120/sglang_triton_dry_run_20260520T101900Z/command.txt`
- FlashInfer decode dry-run command:
  `results/repros/sglang_attention_backend_sm120/sglang_flashinfer_decode_dry_run_20260520T101901Z/command.txt`

### Classification

`blocked: missing local SGLang runtime`

### Next action

Install or check out SGLang in an isolated environment without downloading large
models, then run static/server-argument tests before any target-model repro.

## 2026-05-20: benchmark matrix real runs blocked by active GPU process

### Goal

Run Stage A one-GPU tiny/small serving benchmarks for vLLM and SGLang.

### Outcome

No real serving benchmark was launched. Guard checks showed both GPUs below the
required `70` GiB free-memory threshold with active `python` PID `2999453` on
both GPUs.

Evidence:

- Status: `results/gpu_status/benchmark_matrix_pre_status.json`
- GPU 0 check: `results/gpu_status/benchmark_matrix_gpu0_check.json`
- GPU 1 check: `results/gpu_status/benchmark_matrix_gpu1_check.json`
- Hardware sentinel status:
  `results/gpu_status/hardware_sentinel_20260520_current_status.json`
- Dry-run matrix:
  `results/benchmarks/20260520T103000Z_stage_a_vllm_tiny_dry_run.summary.json`
  and
  `results/benchmarks/20260520T103001Z_stage_a_sglang_tiny_dry_run.summary.json`

### Classification

`blocked: gpu contention`

### Next action

Retry Stage A only after a fresh guard check passes:

```bash
python scripts/gpu_guard.py check --gpus 1 --min-free-gb 70
```

## 2026-05-20: post-queue audit confirms benchmarks still blocked

### Goal

Audit current evidence without launching new heavy workloads.

### Outcome

No new GPU-heavy workload was launched. A fresh GPU status check still showed
both GPUs occupied by active `python` PID `2999453` and below the required
70 GiB free-memory threshold.

Evidence:

- Audit status: `results/gpu_status/post_queue_audit_status.json`
- Evidence ledger: `docs/evidence_ledger.md`

### Classification

`blocked: gpu contention`

### Next action

Wait for a clean GPU window, then rerun only the one-GPU Stage A smoke command
after:

```bash
python scripts/gpu_guard.py check --gpus 1 --min-free-gb 70
```

## 2026-05-23: reproducibility test pass initially exposed optional vLLM import brittleness

### Goal

Run the lightweight local test suite after adding reproducibility scripts.

### Outcome

The first full `pytest -q` attempt reported one failure in the new
reproducibility tests and then spent several minutes inside the optional vLLM
static-probe subprocess. No GPU workload was launched.

Fixes made:

- `scripts/collect_versions.py` now includes `HF_` and `HUGGINGFACE_`
  environment variables in sanitized output while redacting token-like names.
- `tests/test_vllm_backend_tools.py` now gates the optional vLLM import/static
  probe behind `BLACKWELL_INFERENCE_RUN_OPTIONAL_IMPORT_TESTS=1` and keeps a
  timeout if explicitly enabled.

Evidence:

- Final test artifact: `results/tests/reproducibility_pytest.txt`

### Classification

`fixed: optional dependency/import brittleness`
