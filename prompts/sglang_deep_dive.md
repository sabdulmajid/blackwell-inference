Focus only on the SGLang RTX6000 Pro Blackwell FP8 attention/backend/shared-memory problem.

Goal:

Produce an upstream-quality SGLang reproducer and patch/issue direction.

Context:

- Public anchor: `sgl-project/sglang#16816` says Qwen3-Next-80B-A3B-Instruct-FP8 on RTX6000 Pro Blackwell had Triton attention shared-memory failure: required around 114688 bytes while hardware limit was around 101376 bytes.
- The issue reports FlashInfer could work with a backend-selection patch.
- Local repro directory: `repros/sglang_attention_backend_sm120/`.

Constraints:

- Do not run GPU-heavy commands without `scripts/run_with_gpu_lock.py`.
- Do not assume FlashInfer is universally correct; validate exact model/backend/version.
- Save requested-vs-hardware shared-memory evidence if Triton fails.
- Save logs and update `TASK_BOARD.md`.

Done when:

- current SGLang attention/backend codepath is mapped,
- a minimal repro command exists,
- the current behavior is tested or a concrete blocker is documented,
- and an upstream PR/issue draft exists under `upstream/sglang/`.
