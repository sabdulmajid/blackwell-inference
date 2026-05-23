# blackwell-inference

`blackwell-inference` is an ML systems investigation into inference behavior on
NVIDIA RTX PRO 6000 Blackwell / SM120 GPUs, focused on the places where modern
serving stacks make hardware-specific decisions that are easy to get wrong.

The project studies two upstream-relevant problems:

- **vLLM FP4 / MXFP4 / NVFP4 backend selection on SM120**
- **SGLang FP8 attention, GDN linear-attention, and GEMM backend behavior on RTX
  PRO Blackwell**

The goal is not to publish premature benchmark numbers. The goal is to build a
disciplined evidence trail: environment metadata, contention state, exact
commands, raw logs, backend-selection traces, correctness checks, and
maintainer-readable issue or patch artifacts.

## Why This Matters

Inference frameworks increasingly route work through specialized kernels:
FlashInfer, CUTLASS, Triton, Marlin, TensorRT-LLM-style paths, fused MoE kernels,
and FP8/FP4 GEMM implementations. Those paths often depend on compute
capability, CUDA version, shared-memory limits, library availability, model
format, and explicit runtime flags.

RTX PRO 6000 Blackwell is a workstation Blackwell GPU with SM120. It should not
be treated casually as identical to datacenter Blackwell. A backend that is
valid on one Blackwell part may be unavailable, slower, or unsafe on another.
This repository exists to turn that ambiguity into reproducible evidence.

## Investigation Tracks

### vLLM: SM120 FP4 / MXFP4 / NVFP4

Public anchor: `vllm-project/vllm#31085`

The investigation asks:

- Which low-precision backend does vLLM choose on SM120?
- When does it fall back to Marlin or emulation?
- Are fallback reasons visible enough for users and maintainers?
- Is an SM120 native path correct, available, and safe to select automatically?

Current source audit:

- Current vLLM upstream inspected at
  `5bb8d2767a2829b56e58c68fa8f380e9e4e2bd3e`.
- Current upstream already contains an explicit SM12x NVFP4 FlashInfer B12x MoE
  expert and an SM120-gated kernel test.
- Installed vLLM `0.12.0` static selector simulation returns `MARLIN` for mocked
  SM120 MXFP4 scenarios. That is source-level evidence only, not runtime proof.

Current conclusion: the next useful vLLM contribution is likely diagnostics and
selector coverage first, not a broad capability-widening patch.

### SGLang: FP8 Attention And Shared-Memory Behavior

Public anchor: `sgl-project/sglang#16816`

The investigation asks:

- Which full-attention backend is selected on SM120?
- Which GDN linear-attention decode and prefill paths are selected?
- Does FP8 GEMM dispatch choose a supported implementation?
- If Triton fails, does the error identify the selected path and requested
  shared-memory shape clearly enough?

Current source audit:

- Current SGLang upstream inspected at
  `a5a64a311a39b153d1e4d3d6bcb67e77cdc9aeae`.
- Current upstream already contains SM120 capability helpers, SM120 Triton
  attention block sizing, and an SM120 FP8 GEMM auto fallback to Triton.
- SGLang is not installed in the current local Python environment, so no local
  SGLang runtime backend claim is made yet.

Current conclusion: the next useful SGLang contribution is precise runtime
diagnostics around selected full-attention, GDN decode, GDN prefill, and FP8 GEMM
paths before changing fallback behavior.

## Evidence So Far

Supported by local artifacts and tests:

- The harness records GPU status, active GPU processes, selected physical GPU
  IDs, and before/after state for GPU-affecting commands.
- Environment collection works without initializing CUDA by default.
- A locked CUDA metadata probe previously confirmed local PyTorch sees an RTX
  PRO 6000 Blackwell GPU with compute capability `(12, 0)`.
- Benchmark dry-runs produce raw JSONL and summaries, and are explicitly marked
  as non-headline evidence.
- Current vLLM and SGLang upstream sources have been inspected in clean sparse
  checkouts.
- Lightweight test suite passes: `29 passed, 1 skipped`.

Not yet claimed:

- no valid real serving benchmark result,
- no throughput, latency, speedup, or regression claim,
- no vLLM runtime backend-selection proof on SM120,
- no SGLang runtime backend-selection proof on SM120,
- no upstream PR-ready behavior-changing patch.

Detailed evidence ledger: `docs/evidence_ledger.md`

## Repository Shape

The repo is organized around evidence and upstream readiness:

- `scripts/` contains safety, environment, version, archive, and GPU-wrapper
  utilities.
- `benchmarks/` contains an OpenAI-compatible benchmark client and summarizer.
- `repros/` contains minimal vLLM and SGLang repro wrappers.
- `docs/` contains the evidence ledger, technical context, benchmark protocol,
  dependency notes, source audit, and failure log.
- `upstream/` contains issue drafts, patch plans, and diff summaries for vLLM
  and SGLang.
- `results/` is for generated local artifacts. Large or machine-specific raw
  outputs are intentionally not committed.

## Current State

This is a serious scaffold and investigation repo, not a finished benchmark
paper. The important work completed so far is:

1. GPU-safe execution policy and lock-wrapper infrastructure.
2. Environment and dependency capture.
3. Benchmark schema and dry-run validation.
4. vLLM and SGLang source-path maps at current upstream commits.
5. Conservative upstream patch plans that separate confirmed facts from
   hypotheses.

The most important blockers are:

- a clean one-GPU runtime window,
- a tiny real model-serving smoke test,
- local SGLang runtime setup in an isolated environment,
- real backend-selection logs,
- correctness checks before any performance comparison.

## Next Milestones

1. Run one valid, uncontended, one-GPU smoke test.
2. Capture vLLM SM120 backend-selection logs with the smallest viable model.
3. Install or check out SGLang in an isolated environment and capture equivalent
   backend logs.
4. Add targeted upstream diagnostics or selector tests.
5. Run only the minimal target repros needed to support an issue update or PR.
6. Publish upstream artifacts once runtime evidence is sufficient.

## Project Standard

No claim without evidence. A result is only treated as real if it has saved raw
artifacts, environment metadata, GPU contention metadata, and a clear validity
label.
