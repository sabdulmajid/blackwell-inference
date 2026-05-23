# vLLM upstream draft: SM120 FP4/MXFP4/NVFP4 backend selection

Status: issue-update draft. Not PR-ready until a locked SM120 runtime repro fills
the hardware evidence sections.

## Summary

I investigated RTX PRO 6000 Blackwell / SM120 backend selection for vLLM
FP4/MXFP4/NVFP4 paths. Static selector simulation of installed vLLM `0.12.0`
returns Marlin on mocked SM120 because native MXFP4 FlashInfer branches are
gated on exact SM100 and the Triton fallback excludes capabilities `>= 11.0`.

Current upstream main has refactored this path into the fused-MoE oracle. It
appears to include an explicit SM120-family allowance for FlashInfer CUTLASS
experts, so I am not proposing a broad capability-widening patch without runtime
evidence.

## Environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- Compute capability: SM120 / `(12, 0)` from locked CUDA device probe
  `results/gpu_runs/20260520T100750Z_sglang_device_probe/verify_blackwell_probe_cuda.json`.
- Installed vLLM: `0.12.0`
- Installed PyTorch: `2.9.0`
- Installed Triton: `3.5.0`
- Installed FlashInfer package: `flashinfer-python 0.5.3`
- Installed vLLM source: Python site-packages `vllm` module path recorded in
  `results/repros/vllm_mxfp4_sm120/static_backend_selection_probe.json`.
- Upstream HEAD observed: `87e31455b056c6ce59bf5dcb3c622155431851db`

Evidence:

- Environment/status: `results/gpu_status/vllm_deep_dive_status_latest.json`
- Latest benchmark-matrix GPU status:
  `results/gpu_status/benchmark_matrix_pre_status.json`
- Static selector probe:
  `results/repros/vllm_mxfp4_sm120/static_backend_selection_probe.json`

## Minimal Reproducer

Static selector probe, no CUDA/model load:

```bash
python scripts/vllm_mxfp4_static_probe.py \
  --out results/repros/vllm_mxfp4_sm120/static_backend_selection_probe.json \
  --upstream-head 87e31455b056c6ce59bf5dcb3c622155431851db
```

Prepared one-GPU runtime repro, not yet run because both GPUs had an active
external compute process:

```bash
python scripts/run_with_gpu_lock.py \
  --gpus 1 \
  --min-free-gb 70 \
  --wait \
  --poll-seconds 30 \
  --max-wait-seconds 600 \
  --label vllm_mxfp4_repro_gpu1 \
  -- \
  bash repros/vllm_mxfp4_sm120/run_repro.sh \
    --model <MXFP4_MODEL_ID_OR_LOCAL_PATH> \
    --tp 1 \
    --dtype auto \
    --quantization mxfp4
```

## Current Behavior

Installed vLLM `0.12.0`, static selector behavior:

- Mocked SM90 + FlashInfer BF16 env flag -> `SM90_FI_MXFP4_BF16`
- Mocked SM100 + FlashInfer CUTLASS env flag -> `SM100_FI_MXFP4_MXFP8_CUTLASS`
- Mocked SM100 + FlashInfer TRTLLM env flag -> `SM100_FI_MXFP4_MXFP8_TRTLLM`
- Mocked SM120 + all FlashInfer MXFP4 env flags + Triton available -> `MARLIN`
- Mocked SM120 + LoRA + all FlashInfer MXFP4 env flags -> `MARLIN`

The SM120 result is from source-level selector simulation only. It does not prove
runtime kernel support or correctness.

## Expected Behavior

Expected behavior should be one of the following, with clear logs:

- Select a native backend only when that backend explicitly supports SM120 and
  the deployment configuration.
- Otherwise fall back safely and explain whether the fallback is due to missing
  package support, rejected kernel configuration, unsupported capability, or an
  intentionally unvalidated SM120 path.

This draft does not assume SM120 is equivalent to SM100.

## Codepath Map

Installed vLLM `0.12.0`:

- `vllm/model_executor/layers/quantization/mxfp4.py`
  - `get_mxfp4_backend_with_lora()`
  - `get_mxfp4_backend(with_lora_support)`
  - `Mxfp4MoEMethod.__init__()`
- `vllm/platforms/interface.py`
  - `DeviceCapability.to_int()`
  - `Platform.has_device_capability()`
  - `Platform.is_device_capability()`
- `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py`
  - `is_flashinfer_fp4_cutlass_moe_available()`
  - `is_flashinfer_fp4_cutedsl_moe_available()`
  - `select_nvfp4_gemm_impl()`
- `vllm/model_executor/layers/quantization/utils/nvfp4_moe_support.py`
  - `detect_nvfp4_moe_support()`
- `vllm/model_executor/layers/quantization/modelopt.py`
  - `ModelOptNvFp4LinearMethod`
  - `ModelOptNvFp4FusedMoE`

Current upstream HEAD:

- `vllm/model_executor/layers/quantization/mxfp4.py`
  delegates routed-expert MXFP4 to `select_mxfp4_moe_backend()`.
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`
  defines `Mxfp4MoeBackend`, priority order, env handling, and backend logging.
- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`
  defines `NvFp4MoeBackend` and NVFP4 backend selection.
- `vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py`
  gates TRTLLM MXFP4 on `is_device_capability_family(100)`.
- `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py`
  explicitly allows SM90, SM100-family, and SM120-family for FlashInfer CUTLASS.

## Root Cause / Hypothesis

Installed `0.12.0` root cause:

- MXFP4 FlashInfer native branches use exact `is_device_capability(100)`.
- SM120 is not exact SM100.
- The Triton fallback is limited to `SM90 <= capability < SM110`.
- The selector therefore reaches Marlin.

Current upstream hypothesis:

- CUTLASS may already have a narrow SM120 allowance.
- TRTLLM/BF16 and some helper branches still use SM100-family checks that exclude
  SM120.
- Runtime evidence is required to determine whether auto-selection picks a
  native CUTLASS path, falls back, or raises for the target model/config.

## Proposed Fix Direction

Diagnostics-first:

1. Improve backend-selection logs to include compute capability, env overrides,
   FlashInfer/CUTLASS/Triton probe results, rejected backend candidates, and
   fallback reason.
2. Add mocked selector tests covering SM90, SM100, SM110, and SM120.
3. Run a hardware-gated SM120 repro before changing capability gates.

If runtime evidence proves a specific SM120 native backend works:

- add an explicit narrow SM120 branch for that backend only,
- keep fallback behavior intact,
- avoid replacing exact SM100 checks with broad `has_device_capability(100)`
  unless the code intentionally supports every capability `>= 10.0`.

## Correctness Validation

Pending locked one-GPU runtime repro. Required before PR:

- deterministic prompt output,
- no textual NaN/Inf in generated output,
- backend log evidence,
- ideally logits or reference-backend comparison for the same prompt/model.

## Performance Validation

No performance claim is made.

Benchmark matrix dry-run artifacts exist only to validate schema:

- `results/benchmarks/20260520T103000Z_stage_a_vllm_tiny_dry_run.summary.json`
- `results/benchmarks/summary.md`

The dry-run rows are not benchmark evidence and are not headline-eligible.

Required before any performance claim:

- raw JSONL:
- summary JSON/CSV/Markdown:
- GPU wrapper metadata:
- contention label:
- token count source:

## Risks

- SM120 may support one FP4 backend but not another.
- `has_device_capability(100)` means `>= 10.0` and can admit SM110/future GPUs.
- Current upstream source and installed vLLM `0.12.0` differ materially.
- Marlin fallback may be correct and safe even if slower.

## Tests

Local repo tests added:

- `tests/test_vllm_backend_tools.py`

Run:

```bash
pytest -q
```
