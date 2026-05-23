# SGLang upstream draft: RTX PRO 6000 Blackwell FP8 attention/backend/shared-memory behavior

Status: investigation draft. Do not submit as a PR until a locked runtime repro
captures SGLang logs, backend selection, and correctness output on this machine.

## Summary

I am investigating Qwen3-Next FP8 backend behavior on an RTX PRO 6000 Blackwell
/ SM120 workstation GPU. The current hypothesis is that failures reported as
Triton shared-memory `OutOfResources` may be caused by either full-attention
Triton extend kernel tile selection or Qwen3-Next hybrid GDN linear-attention
decode/prefill defaults, which are selected separately from
`--attention-backend`.

Current upstream already has SM120-specific logic in Triton extend attention and
Blackwell-aware FP8 GEMM dispatch. I do not yet have enough runtime evidence to
propose a broad hardware-gate change.

## Environment

Evidence from local locked GPU metadata probe:

- Artifact: `results/gpu_runs/20260520T100750Z_sglang_device_probe/verify_blackwell_probe_cuda.json`
- Wrapper metadata: `results/gpu_runs/20260520T100750Z_sglang_device_probe/run_meta.json`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`
- Compute capability: `(12, 0)`
- Total memory: `101971591168` bytes
- PyTorch-reported `shared_memory_per_block`: `49152`
- PyTorch-reported `shared_memory_per_multiprocessor`: `102400`
- PyTorch: `2.9.0+cu128`
- CUDA from PyTorch: `12.8`
- Triton: `3.5.0`
- FlashInfer: `0.5.3`
- Local `sglang`: not installed
- Local `sgl-kernel`: not installed
- SGLang source map previously inspected: upstream `main`
  `47979fb252ce0954d1076c67183879bb52e17476`
- Current remote `main` observed during patch-discipline pass:
  `1bd4f94598a621cf5e8c27686311e92134e9edb0`; revalidate source paths after
  checkout before editing or posting.

## Reproduction Command

Triton path to capture the suspected failure:

```bash
python scripts/run_with_gpu_lock.py --gpus 0 --min-free-gb 70 --wait \
  --label sglang_fp8_triton -- \
  bash repros/sglang_attention_backend_sm120/run_repro.sh \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --linear-attn-decode-backend triton \
    --linear-attn-prefill-backend triton
```

Fallback experiment, if the Triton repro fails in GDN decode but prefill remains
Triton-only on SM120:

```bash
python scripts/run_with_gpu_lock.py --gpus 0 --min-free-gb 70 --wait \
  --label sglang_fp8_flashinfer_decode -- \
  bash repros/sglang_attention_backend_sm120/run_repro.sh \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
    --attention-backend flashinfer \
    --fp8-gemm-backend flashinfer_cutlass \
    --linear-attn-decode-backend flashinfer \
    --linear-attn-prefill-backend triton
```

These commands have not been run against the large public target model in this
pass because SGLang is not installed locally and the investigation has not yet
exhausted static/small-model paths.

## Expected Behavior

On SM120, SGLang should either:

- select a supported attention and linear-attention backend combination, or
- fail early with a clear message that includes the selected full-attention
  backend, selected linear-attention decode/prefill backends, queried device
  properties, Triton-reported requested/limit bytes when available, and
  actionable fallback guidance.

For Qwen3-Next GDN, a safe fallback should not imply FlashInfer prefill support
on SM120 unless that support is available and tested.

## Actual Behavior

Runtime actual behavior is pending a locked SGLang repro. The local harness now
captures these artifacts for the next run:

- `verify_blackwell.json`
- `command.txt`
- `server_stdout.log`
- `server_stderr.log`
- `bench_raw.jsonl`
- `bench_summary_stdout.json`
- `backend_grep.txt`
- `backend_summary.json`
- `status.txt`

`backend_summary.json` extracts:

- selected full-attention backend if logged,
- selected GDN/linear-attention decode and prefill backends,
- FP8 GEMM backend mentions,
- Triton `OutOfResources` and shared-memory requested/limit bytes when present,
- failure-class labels such as `triton_out_of_resources_shared_memory`.

## Codepath Map

- `python/sglang/srt/server_args.py`
  - `ServerArgs.__post_init__`
  - `_handle_model_specific_adjustments`
  - `_get_default_attn_backend`
  - `_handle_attention_backend_compatibility`
- `python/sglang/srt/utils/common.py`
  - `is_sm120_supported`
  - `is_blackwell_supported`
  - `get_device_capability`
- `python/sglang/srt/model_executor/model_runner.py`
  - `init_attention_backend`
  - `_get_attention_backend`
  - `_get_attention_backend_from_str`
- `python/sglang/srt/layers/attention/attention_registry.py`
  - `ATTENTION_BACKENDS`
  - `create_triton_backend`
  - `create_flashinfer_backend`
  - `attn_backend_wrapper`
- `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
  - `_get_block_sizes_for_extend_attention`
  - `extend_attention_fwd`
  - `extend_attention_fwd_unified`
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
- `python/sglang/srt/layers/quantization/fp8_utils.py`
  - `initialize_fp8_gemm_config`
  - `_dispatch_auto_backend`
  - `_dispatch_explicit_backend`

## Suspected Root Cause

For Qwen3-Next FP8 on RTX PRO 6000 Blackwell, a shared-memory failure may be in
one of two independently selected paths:

1. Full-attention Triton extend kernels choose a block shape that exceeds the
   target device's available shared memory.
2. Hybrid GDN linear-attention defaults select Triton for decode/prefill and a
   lower-level Triton/FLA GDN kernel exceeds the shared-memory limit.

`--attention-backend flashinfer` alone is not enough evidence that the GDN
linear-attention path avoided Triton. The linear-attention flags must be logged
and tested.

## Proposed Patch Direction

Patch only after runtime evidence identifies the failing path.

If full-attention Triton extend attention fails:

- refine the SM120 block-size selection in
  `triton_ops/extend_attention.py::_get_block_sizes_for_extend_attention`,
- log queried device shared-memory properties and selected block sizes, and
- add a unit test for SM120 block-size choice.

If GDN linear attention fails:

- preserve Triton prefill unless FlashInfer prefill support is proven for SM120,
- allow or recommend FlashInfer GDN decode only when FlashInfer support and
  device checks pass,
- improve the error text for Triton GDN shared-memory `OutOfResources`, and
- add a dispatch test for
  `--linear-attn-decode-backend flashinfer --linear-attn-prefill-backend triton`.

If FP8 GEMM selection is implicated:

- keep the GEMM patch separate from attention/backend selection,
- do not replace SM120 `auto -> triton` without correctness and performance
  evidence for the replacement backend.

## Correctness Validation

Pending. The next runtime repro must save:

- prompt text,
- generated text,
- token counts,
- NaN/Inf or error checks if logits/tensors are exposed,
- raw OpenAI-compatible benchmark JSONL if the server reaches ready state.

## Performance Validation

No performance claim is made. Benchmark matrix dry-run artifacts exist only to
validate schema:

- `results/benchmarks/20260520T103001Z_stage_a_sglang_tiny_dry_run.summary.json`
- `results/benchmarks/summary.md`

The dry-run rows are not benchmark evidence and are not headline-eligible.

Before any performance claim, the evidence must include:

- raw JSONL,
- summary JSON/CSV/Markdown,
- GPU wrapper metadata,
- contention label,
- token count source,
- exact selected GPU IDs.

## Risks

- FlashInfer support is model-path-specific. Current upstream comments indicate
  GDN FlashInfer on SM100+ is decode-only.
- Local FlashInfer is `0.5.3`, while current upstream comments mention newer
  FlashInfer versions for GDN support.
- Triton shared-memory failures may require kernel-specific tuning, not only
  backend selection.
- SM120 should not be treated as interchangeable with SM100 without evidence.
