# SGLang SM120 FP8 Attention Backend Investigation

Status: diagnostics-ready; not PR-ready until a locked one-GPU SGLang runtime
repro captures backend logs and correctness output.

## Source State

- Local `external/sglang`: absent.
- Installed `sglang`: absent.
- Installed `sgl-kernel`: absent.
- Installed `flashinfer-python`: `0.5.3`.
- Installed `torch`: `2.9.0+cu128`.
- Installed `triton`: `3.5.0`.
- Upstream SGLang source map previously inspected via GitHub at commit
  `47979fb252ce0954d1076c67183879bb52e17476`.
- Current remote `main` observed during the patch-discipline pass:
  `1bd4f94598a621cf5e8c27686311e92134e9edb0`; revalidate source paths after
  checkout before editing.

Bootstrap command for a future source checkout:

```bash
mkdir -p external
git clone --filter=blob:none --sparse https://github.com/sgl-project/sglang.git external/sglang
git -C external/sglang sparse-checkout set \
  python/sglang/srt/server_args.py \
  python/sglang/srt/model_executor \
  python/sglang/srt/layers/attention \
  python/sglang/srt/layers/quantization \
  python/sglang/srt/configs \
  python/sglang/srt/models \
  test
git -C external/sglang checkout 1bd4f94598a621cf5e8c27686311e92134e9edb0
git -C external/sglang rev-parse HEAD
```

## Local SM120 Device Evidence

Device properties were collected through the GPU lock wrapper with PyTorch CUDA
probing enabled:

```bash
python scripts/run_with_gpu_lock.py --gpus 0 --min-free-gb 70 --wait \
  --max-wait-seconds 300 \
  --label sglang_device_probe -- \
  bash -lc 'python scripts/verify_blackwell.py --probe-cuda --out "$BLACKWELL_INFERENCE_GPU_RUN_DIR/verify_blackwell_probe_cuda.json"'
```

Artifacts:

- `results/gpu_runs/20260520T100750Z_sglang_device_probe/run_meta.json`
- `results/gpu_runs/20260520T100750Z_sglang_device_probe/verify_blackwell_probe_cuda.json`

Observed through PyTorch in that run:

- GPU name: `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`
- Capability: `(12, 0)`
- Visible device count: `1`
- Total memory: `101971591168` bytes
- SM count: `188`
- `shared_memory_per_block`: `49152`
- `shared_memory_per_multiprocessor`: `102400`

This local value is evidence for this machine only. It should not be replaced
with a hardcoded RTX PRO 6000 shared-memory constant in a patch.

## Codepath Map

### CLI and Backend Selection

- `python/sglang/srt/server_args.py`
  - `ServerArgs.__post_init__`
  - `_handle_model_specific_adjustments`
  - `_get_default_attn_backend`
  - `_handle_attention_backend_compatibility`
  - `_handle_mamba_radix_cache`

These functions normalize CLI flags such as `--attention-backend`,
`--fp8-gemm-backend`, and the linear-attention backend flags before model
runner initialization.

### Device Capability Helpers

- `python/sglang/srt/utils/common.py`
  - `get_device_sm`
  - `get_device_capability`
  - `is_sm100_supported`
  - `is_sm120_supported`
  - `is_blackwell_supported`

Patch plans should use these helpers for coarse capability checks, but runtime
shared-memory behavior still needs actual device property logging.

### Attention Registry and Model Runner

- `python/sglang/srt/layers/attention/attention_registry.py`
  - `ATTENTION_BACKENDS`
  - `create_triton_backend`
  - `create_flashinfer_backend`
  - `create_trtllm_mha_backend`
  - `attn_backend_wrapper`
- `python/sglang/srt/model_executor/model_runner.py`
  - `init_attention_backend`
  - `_get_attention_backend`
  - `_get_attention_backend_from_str`

`attn_backend_wrapper` is the key hybrid-model handoff point. For hybrid GDN
models on Blackwell, current upstream permits full attention backends including
`triton` and `flashinfer`, then wraps the chosen full-attention backend with a
linear-attention backend.

### Triton Attention

- `python/sglang/srt/layers/attention/triton_backend.py`
  - `TritonAttnBackend`
- `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`
  - `_get_block_sizes_for_extend_attention`
  - `extend_attention_fwd`
  - `extend_attention_fwd_unified`

Current upstream includes an SM120 branch in `extend_attention.py` that appears
intended to reduce shared-memory pressure for RTX PRO 6000-class devices. This
still needs runtime evidence on the target machine and target model path.

### Qwen3-Next Hybrid GDN Linear Attention

- `python/sglang/srt/configs/qwen3_next.py`
  - `Qwen3NextConfig`
  - `layers_block_type`
  - `linear_layer_ids`
  - `full_attention_layer_ids`
- `python/sglang/srt/models/qwen3_next.py`
  - `Qwen3GatedDeltaNet`
  - `Qwen3HybridLinearDecoderLayer`
  - `Qwen3HybridAttentionDecoderLayer`
  - `Qwen3NextForCausalLM`
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

The GDN FlashInfer kernel comments and guards indicate SM100+ support is
decode-only. Prefill should remain Triton on SM120 unless upstream changes that
support matrix.

### FP8 GEMM Backend

- `python/sglang/srt/layers/quantization/fp8_utils.py`
  - `initialize_fp8_gemm_config`
  - `_dispatch_auto_backend`
  - `_dispatch_explicit_backend`
  - `_get_flashinfer_groupwise_backend`

Current upstream forces `fp8_gemm_runner_backend=auto` to `triton` on SM120.
Explicit `flashinfer_cutlass` is accepted through Blackwell checks when
FlashInfer is available. This is a separate decision from full-attention and
linear-attention backend selection.

## Root-Cause Hypothesis

The target failure is narrowed, not reproduced locally in this pass.

For Qwen3-Next FP8 on RTX PRO 6000 Blackwell, a failure that reports Triton
`OutOfResources` with required shared memory larger than the hardware limit can
come from either:

1. full-attention Triton extend kernels choosing a tile/block shape that exceeds
   the local SM120 shared-memory limit, or
2. the Qwen3-Next hybrid GDN linear-attention path defaulting decode and prefill
   to Triton, with an FLA/Triton GDN kernel exceeding the local limit.

The second path is important because `--attention-backend flashinfer` alone does
not necessarily move GDN linear attention away from Triton. The linear attention
flags are separate:

- `--linear-attn-decode-backend`
- `--linear-attn-prefill-backend`
- `--linear-attn-backend`

A safe initial fallback experiment is FlashInfer for GDN decode and Triton for
GDN prefill:

```bash
--attention-backend flashinfer \
--fp8-gemm-backend flashinfer_cutlass \
--linear-attn-decode-backend flashinfer \
--linear-attn-prefill-backend triton
```

This is a hypothesis, not a proven fix.

## Reproducer State

The local repro script is parameterized and refuses unlocked GPU launches:

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

The script writes:

- `verify_blackwell.json`
- `command.txt`
- `server_stdout.log`
- `server_stderr.log`
- `bench_raw.jsonl`
- `bench_summary_stdout.json`
- `backend_grep.txt`
- `backend_summary.json`
- `status.txt`

`backend_summary.json` is produced by
`scripts/extract_sglang_backend_evidence.py` and extracts selected backend
lines, GDN decode/prefill choices, Triton shared-memory requested/limit bytes,
and failure classes.

## Minimal Patch Plan

Do not patch external SGLang until a runtime repro provides a precise failure
site.

If the failure is in `triton_ops/extend_attention.py`:

- preserve or refine the SM120-specific block-size branch,
- log the selected block sizes and queried device shared-memory properties, and
- add a targeted test for SM120 block-size selection.

If the failure is in GDN linear attention:

- keep FlashInfer GDN decode behind the existing availability and device checks,
- keep SM120 prefill on Triton unless FlashInfer prefill support is proven,
- improve error text or fallback guidance when Triton GDN hits shared-memory
  `OutOfResources`, and
- add tests that `--linear-attn-decode-backend flashinfer` plus
  `--linear-attn-prefill-backend triton` dispatches as intended on Blackwell.

If the failure is in FP8 GEMM:

- keep GEMM selection separate from attention selection,
- do not replace SM120 `auto -> triton` without correctness and performance
  evidence for the replacement backend.

## Next Evidence Needed

1. Install or check out SGLang without downloading large models.
2. Run a static/server-argument test first, if available.
3. Run the locked one-GPU repro against the smallest available Qwen3-Next/GDN
   fixture or checkpoint that exercises backend dispatch.
4. If the large public FP8 model is the only path that exercises the failure,
   document that explicitly before requesting approval to download/use it.
