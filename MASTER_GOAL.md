# Master Goal

Build `blackwell-inference`, an upstream-quality SM120 Blackwell inference-systems harness focused on:

1. vLLM SM120 FP4/MXFP4/NVFP4 backend selection.
2. SGLang RTX6000 Pro Blackwell FP8 attention/backend/shared-memory behavior.

The project is complete only when every acceptance criterion below is satisfied or explicitly marked as blocked with evidence.

## Global acceptance criteria

- [ ] Repo has a clean, reproducible setup path.
- [ ] `python scripts/verify_blackwell.py --out results/env/verify_blackwell.json` works.
- [ ] `python scripts/gpu_guard.py status` works.
- [ ] Every GPU-consuming script uses `scripts/run_with_gpu_lock.py` or documents why it is read-only/no-GPU.
- [ ] Environment metadata is captured for every run.
- [ ] Benchmark outputs are saved as JSONL and summarized as CSV/Markdown.
- [ ] Logs preserve exact commands, commits, model IDs/paths, dtypes, backends, CUDA_VISIBLE_DEVICES, and GPU status before/after.
- [ ] `TASK_BOARD.md` is up to date.
- [ ] No claim of speedup/correctness exists without linked evidence.
- [ ] A final review pass identifies risks, unverified claims, and next work.

## vLLM acceptance criteria

- [ ] Current vLLM `main` or selected commit is checked out and recorded.
- [ ] Relevant codepaths for MXFP4/NVFP4/FP4 backend selection are mapped.
- [ ] Current behavior on RTX PRO 6000 Blackwell / SM120 is tested.
- [ ] Minimal reproducer exists under `repros/vllm_mxfp4_sm120/`.
- [ ] Reproducer logs backend selected and whether fallback occurs.
- [ ] Correctness check exists: deterministic prompt, logits comparison where feasible, or output sanity + NaN/Inf checks.
- [ ] Benchmark before/after exists if a patch is made.
- [ ] Patch candidate is minimal and maintainable, or the issue is closed/upstream fixed and the repo documents that fact.
- [ ] Upstream PR or issue comment draft exists under `upstream/vllm/`.

## SGLang acceptance criteria

- [ ] Current SGLang `main` or selected commit is checked out and recorded.
- [ ] Attention backend / linear attention / hybrid GDN backend codepaths are mapped.
- [ ] Current behavior on RTX PRO 6000 Blackwell / SM120 is tested.
- [ ] Minimal reproducer exists under `repros/sglang_attention_backend_sm120/`.
- [ ] Reproducer logs backend selected, hardware shared-memory limit, requested shared-memory size if failure occurs, and relevant stack trace.
- [ ] Correctness check exists: deterministic prompt, output sanity, NaN/Inf checks, and mismatch notes.
- [ ] Benchmark Triton vs FlashInfer / allowed backend options where feasible.
- [ ] Patch candidate is minimal and maintainable, or the issue is closed/upstream fixed and the repo documents that fact.
- [ ] Upstream PR or issue comment draft exists under `upstream/sglang/`.

## Benchmark acceptance criteria

- [ ] At least one small smoke model benchmark succeeds.
- [ ] At least one large-model or target-model repro attempt is recorded, even if blocked by model access or install constraints.
- [ ] Single-GPU and 2-GPU modes are tested where feasible.
- [ ] TTFT, TPOT, output tokens/sec, requests/sec, GPU memory, backend, dtype, context length, concurrency, and success/failure are logged.
- [ ] Benchmark excludes runs where unrelated GPU users/processes were active, unless explicitly marked `contended=true`.
- [ ] Warmup count and repeated-trial count are recorded.
- [ ] Results summarize median and P95 for latency metrics.

## Upstream-quality acceptance criteria

- [ ] Minimal repro can be copied by a maintainer.
- [ ] Patch does not overgeneralize SM100 and SM120 behavior without evidence.
- [ ] Patch includes tests or explains why test coverage is only manual/hardware-gated.
- [ ] PR summary includes before/after behavior.
- [ ] PR summary includes environment metadata.
- [ ] PR summary includes correctness evidence.
- [ ] PR summary includes performance evidence if performance is claimed.

## Stopping policy

Do not stop merely because one run succeeds. Stop only when:

1. all acceptance criteria are met; or
2. a criterion is blocked by a concrete external limitation, documented with evidence and a next action.

Examples of valid blockers:

- target model requires gated access and access is unavailable,
- current upstream main already fixed the problem and no patch is needed,
- current framework install fails because of a documented upstream dependency issue,
- GPUs are unavailable due to other users/processes during the configured benchmark window.

Invalid blockers:

- “Could not reproduce” without exact command/logs,
- “probably fixed” without testing current main,
- “benchmark looked good” without saved JSONL/CSV evidence,
- “GPU busy” without `nvidia-smi`/guard logs.
