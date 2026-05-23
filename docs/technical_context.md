# Technical Context

## SM120 vs SM100 caution

RTX PRO 6000 Blackwell / SM120 is not automatically equivalent to datacenter Blackwell SM100/B100/B200. The project must verify kernel support and hardware limits empirically.

Specific risks:

- backend selection helpers may only recognize SM100 family,
- kernels tuned for B100/B200 may assume larger shared memory or TMA features,
- FP4/NVFP4/MXFP4 support may depend on library versions,
- FlashInfer/CUTLASS/Triton kernels may differ in support matrix,
- MoE kernels may fail while dense kernels work,
- a fallback backend may be correct but slower.

## vLLM concepts to map

- quantization config classes,
- backend selection helpers,
- current platform capability helpers,
- Marlin fallback,
- FlashInfer/CUTLASS native low-precision paths,
- MoE GEMM backend selection,
- tests for backend-selection behavior,
- OpenAI-compatible server flags.

Current vLLM investigation note:

- `docs/vllm_sm120_backend_selection.md` maps installed vLLM `0.12.0`
  plus current upstream source inspected at
  `5bb8d2767a2829b56e58c68fa8f380e9e4e2bd3e`.
- Static evidence for installed vLLM is saved at
  `results/repros/vllm_mxfp4_sm120/static_backend_selection_probe.json`.
- Installed vLLM's legacy MXFP4 selector returns Marlin for mocked SM120 even
  when FlashInfer and Triton availability probes are forced true, because exact
  SM100 checks fail and the Triton capability range excludes SM120.
- Current upstream main has an explicit SM12x FlashInfer B12x NVFP4 MoE expert,
  an SM120-gated kernel test, and a `flashinfer_b12x` explicit backend option.
  The next necessary evidence is a locked one-GPU runtime repro rather than a
  broad selector patch.

## SGLang concepts to map

- attention backend flags,
- linear-attention backend flags,
- hybrid GDN model config handling,
- Triton backend shared-memory usage,
- FlashInfer backend acceptance logic,
- FP8/FP4 GEMM backend flags,
- server launch path and model runner selection.

Current SGLang investigation note:

- `docs/sglang_sm120_attention_backend.md` maps an earlier source snapshot;
  `docs/current_upstream_source_audit.md` revalidates current upstream source at
  `a5a64a311a39b153d1e4d3d6bcb67e77cdc9aeae`.
- `external/sglang` now exists as a clean sparse checkout, but the `sglang`
  package is still absent from the current Python environment, so no runtime
  SGLang backend claim has been made.
- Local SM120 device properties were collected under the GPU lock wrapper at
  `results/gpu_runs/20260520T100750Z_sglang_device_probe/verify_blackwell_probe_cuda.json`.
- PyTorch reported compute capability `(12, 0)` and
  `shared_memory_per_multiprocessor=102400` for visible GPU 0 in that locked
  probe.
- Current upstream contains SM120 capability helpers, SM120 Triton attention
  block sizing for smaller workstation shared-memory limits, and an SM120 FP8
  GEMM auto fallback to Triton.
- Qwen3-Next hybrid GDN models have separate full-attention and linear-attention
  backend selection. `--attention-backend flashinfer` alone does not prove the
  GDN path avoided Triton.
- FlashInfer GDN code comments and guards in current upstream indicate SM100+
  decode-only support; prefill should remain Triton unless newer evidence says
  otherwise.

## Benchmark concepts

- TTFT: time until first streamed token.
- TPOT: time per output token after the first token.
- Prefill: processing prompt tokens; compute-heavy and context-length sensitive.
- Decode: generating one token at a time; KV-cache and memory-bandwidth sensitive.
- KV cache: per-layer K/V tensors stored for generated sequences.
- Tensor parallelism: shards model layers across GPUs; can add PCIe/NCCL communication overhead.
- Contention: unrelated GPU users/processes invalidating performance measurements.

Current benchmark status:

- `docs/benchmark_matrix.md` records the staged benchmark plan and current
  blocker.
- `results/benchmarks/summary.md` currently contains dry-run rows only.
- `benchmarks/serve_bench.py` now requests streamed usage metadata by default so
  future OpenAI-compatible runs can produce exact token counts when supported.
- `benchmarks/summarize_results.py` reads wrapper `gpu_before.json` and
  `gpu_after.json` when present and keeps low-sample, approximate-token,
  dry-run, contended, and unlocked rows out of headline metrics.
