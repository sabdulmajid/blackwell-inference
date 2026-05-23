# Project Brief: blackwell-inference

## Thesis

2x NVIDIA RTX PRO 6000 Blackwell GPUs with 96GB VRAM each are unusually strong hardware for local LLM inference-systems research. The highest-value use is not training a large model from scratch. The highest-value use is to debug and benchmark the software stack that frontier labs care about:

- low-precision inference,
- FP8 / FP4 / MXFP4 / NVFP4 numerics,
- kernel backend selection,
- attention backend behavior,
- KV-cache pressure,
- tensor parallelism over PCIe,
- correctness under quantized kernels,
- and robust reproducibility across framework commits.

The two project anchors are deliberately upstream-facing:

1. vLLM SM120 native low-precision backend selection.
2. SGLang RTX6000 Pro Blackwell attention/backend/shared-memory behavior.

The project should produce artifacts that can be understood by maintainers and by frontier-lab interviewers.

## Problem 1: vLLM SM120 FP4/MXFP4/NVFP4 backend selection

Known public anchor:

- `vllm-project/vllm#31085`: SM120 / RTX 6000 Pro Blackwell compute capability `(12, 0)` reportedly not recognized in MXFP4 backend selection, causing fallback to Marlin instead of native NVFP4 kernels.
- Issue root-cause claim: `vllm/model_executor/layers/quantization/mxfp4.py` checks SM100 family, but SM120 has major version 12 and does not match `is_device_capability_family(100)`.

Related search targets:

- `vllm-project/vllm#23497` — FP4 not leveraged on RTX 6000 Pro Blackwell.
- `vllm-project/vllm#30135` — MXFP4 still falling back to Marlin for RTX PRO 6000 / SM120.
- `vllm-project/vllm#32826` — MiniMax NVFP4 issues on dual RTX PRO 6000 Blackwell.
- `vllm-project/vllm#33416` — NVFP4 MoE kernels fail on RTX Blackwell SM12.0.

Important technical caveat:

Do not assume SM120 can use every SM100/B200 kernel. RTX Blackwell SM120 can differ from datacenter Blackwell SM100 in kernel requirements, shared memory, Tensor Memory Accelerator behavior, supported codepaths, or runtime libraries. A correct fix may be backend selection, fallback logic, clearer errors, or dedicated SM120 capability checks—not blindly widening SM100 checks.

## Problem 2: SGLang RTX6000 Pro Blackwell FP8 attention backend/shared-memory behavior

Known public anchor:

- `sgl-project/sglang#16816`: Qwen3-Next-80B-A3B-Instruct-FP8 on RTX6000 Pro Blackwell reports problems.
- The issue says Triton attention backend assumed server-size shared memory around 114k, while RTX6000 Pro hardware limit is around 99k/101376 bytes.
- It reports FlashInfer could run in a newer update but required a patch to attention backend acceptance logic for a hybrid GDN model.

Related search targets:

- `sgl-project/sglang#18954` — NVFP4 models produce NaN outputs on RTX PRO Blackwell.
- Current SGLang docs for attention backend and linear-attention backend options.
- Current backend registry / runner / hybrid GDN config code.

Important technical caveat:

Do not treat “FlashInfer works” as a universal fix. The artifact must document exactly which model, checkpoint format, quantization, backend, SGLang commit, FlashInfer version, driver, CUDA, PyTorch, Triton, and GPU limit were used.

## Why this matters for frontier labs

This project hits the real constraints frontier labs care about:

- Hardware-specific inference runtime correctness.
- Low-precision kernel selection.
- Quantized MoE and attention execution.
- Regression-quality minimal repros.
- Benchmark methodology.
- Debugging frameworks used by the broader AI ecosystem.
- Running large models on constrained multi-GPU workstation hardware.

## Non-goals

- Do not train a large model from scratch.
- Do not build a demo chatbot UI.
- Do not chase synthetic tokens/sec without correctness checks.
- Do not submit speculative PRs without reproducer, tests, logs, and benchmark evidence.
- Do not run unguarded multi-GPU benchmarks that can interfere with other users.

## Final external artifacts

1. Public repo with reproducible scripts.
2. Two technical writeups:
   - `RTX PRO 6000 Blackwell: vLLM FP4/MXFP4 Backend Selection and Benchmark Notes`
   - `SGLang FP8 on RTX PRO 6000 Blackwell: Attention Backend, Shared Memory, and FlashInfer Fallbacks`
3. Upstream PRs or issue updates.
4. Resume bullets with exact results.

## Current status

The current repository has the safety harness, environment verifier, benchmark
client, dry-run matrix, codepath maps, and upstream draft structure in place.
Runtime benchmark/repro evidence is still blocked by GPU contention and missing
SGLang local runtime. Current dry-run benchmark rows are not performance
evidence.
