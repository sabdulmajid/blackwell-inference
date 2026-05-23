Focus only on the vLLM SM120 FP4/MXFP4/NVFP4 problem.

Goal:

Produce an upstream-quality vLLM reproducer and patch/issue direction.

Context:

- Public anchor: `vllm-project/vllm#31085` says SM120 / RTX 6000 Pro Blackwell compute capability `(12,0)` is not recognized in MXFP4 backend selection and falls back to Marlin.
- Related issues: `#23497`, `#30135`, `#32826`, `#33416`.
- Local repro directory: `repros/vllm_mxfp4_sm120/`.

Constraints:

- Do not run GPU-heavy commands without `scripts/run_with_gpu_lock.py`.
- Do not widen SM100 checks to SM120 unless evidence supports the target backend.
- Prefer reproducibility and diagnostics over speculative patching.
- Save logs and update `TASK_BOARD.md`.

Done when:

- current vLLM backend-selection codepath is mapped,
- a minimal repro command exists,
- the current behavior is tested or a concrete blocker is documented,
- and an upstream PR/issue draft exists under `upstream/vllm/`.
