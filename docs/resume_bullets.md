# Evidence-Backed Resume Bullets

These bullets avoid performance claims because no valid uncontended real serving
benchmark exists yet.

- Built a reproducible Blackwell inference-systems harness for 2x RTX PRO 6000
  SM120 GPUs, including GPU contention detection, per-GPU lock enforcement,
  environment capture, raw JSONL benchmark output, and summary validity gating.
- Mapped vLLM `0.12.0` and current-upstream FP4/MXFP4/NVFP4 backend-selection
  paths, with a static SM120 selector probe showing installed MXFP4 fallback to
  Marlin under mocked SM120 conditions.
- Mapped SGLang current-upstream FP8 attention, FlashInfer, Triton, Qwen3-Next
  hybrid GDN, and FP8 GEMM backend paths, and added structured log extraction
  for backend selection and Triton shared-memory failures.
- Added benchmark methodology safeguards that exclude dry-run, contended,
  unlocked, approximate-token, and low-sample runs from headline metrics.

Do not claim tokens/sec, speedup, correctness, or backend validity until a
`valid_uncontended` real benchmark/repro exists under `results/`.
