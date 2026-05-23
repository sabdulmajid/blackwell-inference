# SGLang Patch Plan: SM120 FP8 Attention / Backend / Shared Memory

Status: not PR-ready. No external SGLang checkout exists in this workspace, so
no upstream branch or source patch was created in this pass.

## Upstream Commit

- Patch target: SGLang `main`
  `1bd4f94598a621cf5e8c27686311e92134e9edb0`
- Prior source map inspected: `47979fb252ce0954d1076c67183879bb52e17476`
- Source checkout: missing (`external/sglang` absent)
- Installed package evidence: `sglang` and `sgl-kernel` absent

Because remote `main` moved since the source map was written, revalidate all
candidate files after checking out the patch target.

## External Repo Status

| Field | Status |
|---|---|
| Path | `external/sglang` |
| Exists | no |
| Remote URL | blocked until checkout; intended `https://github.com/sgl-project/sglang.git` |
| Current branch | none |
| Commit SHA | none locally; remote target `1bd4f94598a621cf5e8c27686311e92134e9edb0` |
| Dirty status | not applicable |
| Untracked files | not applicable |
| Existing diff | none |
| Relevant tests | blocked until checkout |

Setup command: `docs/external_repo_setup.md`.

## Files And Functions Implicated

Backend and CLI selection:

- `python/sglang/srt/server_args.py`
  - `ServerArgs.__post_init__`
  - `_handle_model_specific_adjustments`
  - `_get_default_attn_backend`
  - `_handle_attention_backend_compatibility`
- `python/sglang/srt/utils/common.py`
  - `get_device_sm`
  - `get_device_capability`
  - `is_sm100_supported`
  - `is_sm120_supported`
  - `is_blackwell_supported`
- `python/sglang/srt/model_executor/model_runner.py`
  - `init_attention_backend`
  - `_get_attention_backend`
  - `_get_attention_backend_from_str`
- `python/sglang/srt/layers/attention/attention_registry.py`
  - `ATTENTION_BACKENDS`
  - `create_triton_backend`
  - `create_flashinfer_backend`
  - `attn_backend_wrapper`

Triton attention:

- `python/sglang/srt/layers/attention/triton_backend.py`
  - `TritonAttnBackend`
- `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
  - `_get_block_sizes_for_extend_attention`
  - `extend_attention_fwd`
  - `extend_attention_fwd_unified`

Hybrid GDN linear attention:

- `python/sglang/srt/layers/attention/linear/utils.py`
  - `initialize_linear_attn_config`
  - `get_linear_attn_decode_backend`
  - `get_linear_attn_prefill_backend`
- `python/sglang/srt/layers/attention/linear/gdn_backend.py`
  - `GDNKernelDispatcher`
  - `GDNAttnBackend`
- `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py`
  - `TritonGDNKernel`
- `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`
  - `FlashInferGDNKernel`

FP8 GEMM:

- `python/sglang/srt/layers/quantization/fp8_utils.py`
  - `initialize_fp8_gemm_config`
  - `_dispatch_auto_backend`
  - `_dispatch_explicit_backend`
  - `_get_flashinfer_groupwise_backend`

## Minimal Patch Objective

Do not hardcode a workaround yet. The first acceptable upstream patch should
improve diagnostics and add targeted dispatch tests:

1. Log selected full-attention backend, linear-attention decode backend,
   linear-attention prefill backend, FP8 GEMM backend, device capability, and
   queried shared-memory limits.
2. When Triton raises shared-memory `OutOfResources`, add context about the
   selected backend path and actionable fallback flags.
3. Add CPU/static tests for backend flag normalization and dispatch decisions,
   especially FlashInfer decode plus Triton prefill for hybrid GDN models.

Only after a locked runtime repro identifies the failing path should a second
patch tune Triton block sizes or change fallback behavior.

## Alternatives Considered

- Treat RTX PRO 6000 Blackwell SM120 as equivalent to server Blackwell.
  Rejected because local evidence shows the machine's queried device properties
  must be logged and used, not assumed.
- Force `--attention-backend flashinfer` for all Qwen3-Next paths.
  Rejected because GDN linear attention has separate decode/prefill backend
  selection and FlashInfer prefill support is not proven.
- Change FP8 GEMM auto selection from Triton to FlashInfer CUTLASS on SM120.
  Rejected because GEMM selection is separate from the suspected attention
  failure and needs correctness/performance evidence.

## Why This Is Not Overbroad

The planned first patch does not remove Triton, does not force FlashInfer, and
does not change all Blackwell devices. It exposes the exact selected path and
adds tests around dispatch choices so a later behavior change can be narrow.

## Evidence Available

- Locked SM120 CUDA device-property probe:
  `results/gpu_runs/20260520T100750Z_sglang_device_probe/verify_blackwell_probe_cuda.json`
- Wrapper metadata for that probe:
  `results/gpu_runs/20260520T100750Z_sglang_device_probe/run_meta.json`
- SGLang dry-run repro commands:
  `results/repros/sglang_attention_backend_sm120/sglang_triton_dry_run_20260520T101900Z/command.txt`
  and
  `results/repros/sglang_attention_backend_sm120/sglang_flashinfer_decode_dry_run_20260520T101901Z/command.txt`
- Evidence ledger:
  `docs/evidence_ledger.md`
- Technical map:
  `docs/sglang_sm120_attention_backend.md`

Confirmed facts:

- Local PyTorch sees compute capability `(12, 0)`.
- Local PyTorch reports `shared_memory_per_multiprocessor=102400`.
- SGLang is not installed locally.
- No real SGLang server repro or benchmark exists yet.

## Evidence Missing

- Checked-out upstream source commit under `external/sglang`.
- SGLang installed or testable in an isolated environment.
- Runtime logs showing the failing backend path.
- Triton requested shared-memory bytes and exact failing kernel.
- Correctness output for a real prompt.
- Any valid benchmark or performance evidence.

## Test Plan

Local harness tests:

```bash
pytest -q
```

After external checkout:

```bash
git -C external/sglang status --short
git -C external/sglang diff --stat
python -m pytest test -q -k "attention_backend or linear_attn or fp8"
```

Run only CPU/static tests unless dependencies are already ready. Any server,
model load, or CUDA probe must go through this repo's GPU lock wrapper.

## PR Risk Assessment

Diagnostics and tests are moderate-to-low risk if log volume is controlled.
Fallback or block-size changes are high risk without a precise runtime failure
artifact, because the suspected failure may be in full attention, GDN linear
attention, or FP8 GEMM.

Reject an SGLang patch if it:

- hardcodes RTX PRO 6000 Blackwell shared-memory constants,
- treats SM120 as server Blackwell without queried device properties,
- mixes full-attention, GDN linear-attention, and FP8 GEMM behavior changes in
  one diff,
- forces FlashInfer for prefill without upstream support and runtime evidence,
- relies on the 80B target model as the only test,
- makes speedup or throughput claims without valid uncontended raw JSONL, or
- removes a fallback path without a targeted test.

## Rollback / Fallback Behavior

The first patch should not alter runtime backend choices. If a later patch adds
fallback behavior, it must preserve explicit user backend overrides and fail
with clear diagnostics when no supported backend exists.

## Submission Readiness

Not ready for PR. Ready to create an upstream branch only after
`external/sglang` is checked out and candidate source paths are revalidated
against `1bd4f94598a621cf5e8c27686311e92134e9edb0`. A behavior-changing patch
needs one locked SGLang runtime repro first.
