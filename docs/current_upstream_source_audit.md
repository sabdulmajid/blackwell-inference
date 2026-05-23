# Current Upstream Source Audit

Audit date: 2026-05-23

This audit used sparse, read-only source checkouts and did not run GPU kernels,
download models, or start servers. A fresh GPU status check during this pass
showed active Python work on both GPUs, so no new runtime benchmark or repro was
attempted.

## External Checkouts

| Project | Path | Commit | State |
|---|---|---:|---|
| vLLM | `external/vllm` | `5bb8d2767a2829b56e58c68fa8f380e9e4e2bd3e` | clean detached sparse checkout |
| SGLang | `external/sglang` | `a5a64a311a39b153d1e4d3d6bcb67e77cdc9aeae` | clean detached sparse checkout |

These directories are local working inputs and are intentionally ignored by the
main repository. They should not be committed into `blackwell-inference`.

## vLLM Findings

Current vLLM `main` already contains explicit SM12x work for NVFP4 MoE:

- `vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py`
  defines `FlashInferB12xExperts`, documented as a FlashInfer CuteDSL fused MoE
  expert for SM12x. Its device guard requires CUDA,
  `current_platform.is_device_capability_family(120)`, and
  `has_flashinfer_b12x_moe()`.
- `tests/kernels/moe/test_flashinfer_b12x_moe.py` is guarded on SM120-family
  devices and FlashInfer B12x kernel availability. It compares the SM12x kernel
  against a BF16 torch MoE reference with FP4 quantization tolerances.
- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` includes
  `NvFp4MoeBackend.FLASHINFER_B12X` and accepts explicit
  `moe_backend="flashinfer_b12x"`, but excludes B12x from automatic selection
  pending the upstream CUTLASS SM121 guard noted in source comments.
- `vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py` still
  gates TRTLLM MXFP4 experts on `is_device_capability_family(100)`, which is
  separate from SM120 and should not be broadened without runtime evidence.
- `vllm/model_executor/layers/quantization/utils/marlin_utils.py` explicitly
  allows SM12x for Marlin W4A8-FP8 input dtype support.

Implication: the likely upstream contribution is no longer a broad "make SM120
known" patch. The current useful patch direction is narrower:

1. improve backend-selection diagnostics so SM120 users can see why B12x,
   CUTLASS, TRTLLM, Marlin, or emulation were accepted or rejected;
2. add selector tests that distinguish SM100-family TRTLLM paths from SM12x
   B12x/CUTLASS/fallback paths; and
3. run a locked runtime repro before proposing any automatic-selection change.

## SGLang Findings

Current SGLang `main` also has explicit SM120 recognition:

- `python/sglang/srt/utils/common.py` defines `is_blackwell_supported()` for
  CUDA device major versions 10, 11, and 12 with CUDA 12.8+, and
  `is_sm120_supported()` for device major version 12 with CUDA 12.8+.
- `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` has an
  SM120 branch in `_get_block_sizes_for_extend_attention()` with smaller block
  choices and a source comment noting RTX PRO 6000 workstation Blackwell has
  smaller shared memory than datacenter Blackwell.
- `python/sglang/srt/server_args.py` intentionally keeps default MHA
  `trtllm_mha` prefill on SM100 only, while allowing `trtllm_mha` decode on
  SM90, SM100, or SM120. The comments say SM120 falls back to FlashInfer in the
  default path.
- `python/sglang/srt/layers/quantization/fp8_utils.py` changes
  `fp8_gemm_runner_backend=auto` to `triton` on SM120, with a TODO about
  verifying CUTLASS once SwapAB is supported.
- Hybrid GDN models still have two separate decisions: the full-attention
  backend in `server_args.py` / `attention_registry.py`, and the linear
  attention decode/prefill backend in
  `python/sglang/srt/layers/attention/linear/utils.py`.
- `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`
  documents FlashInfer GDN support as decode-only on SM100+; prefill and target
  verify remain Triton/fallback paths.

Implication: the SGLang patch should not simply add SM120 capability checks.
Current source already contains several. The meaningful next patch direction is
diagnostics and exact fallback evidence:

1. log selected full-attention, GDN decode, GDN prefill, and FP8 GEMM backends
   together with capability and queried shared-memory properties;
2. add static tests for SM120 backend normalization and explicit user override
   behavior; and
3. only tune Triton block sizes or fallback behavior after a locked runtime log
   identifies the failing kernel and requested shared-memory size.

## Evidence Status

Proven in this pass:

- current vLLM and SGLang upstream source can be checked out sparsely and
  inspected without model downloads;
- both external checkouts are clean;
- both upstreams now contain explicit SM120-specific code paths.

Not proven in this pass:

- any vLLM or SGLang runtime backend selection on this machine;
- correctness of any FP4/FP8 path;
- throughput, latency, speedup, or regression;
- whether current upstream already fixes the previously suspected behavior.

## Next Evidence Target

The next impactful result should be one valid, locked, one-GPU runtime smoke
test when a GPU is uncontended:

```bash
python scripts/gpu_guard.py check --gpus 0 --min-free-gb 70
python scripts/run_with_gpu_lock.py --gpus 0 --min-free-gb 70 --wait --label vllm_backend_smoke -- <small locked repro command>
```

If GPU 0 is not eligible but GPU 1 is, use GPU 1 and record the selection
rationale. Do not run the command while unrelated active GPU processes are
present on the selected GPU.
