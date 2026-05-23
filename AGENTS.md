# AGENTS.md

You are working on `blackwell-inference`, a multi-day LLM inference-systems project targeting 2x NVIDIA RTX PRO 6000 Blackwell / SM120 GPUs.

## Mission

Build an upstream-quality project around two concrete problems:

1. vLLM SM120 FP4/MXFP4/NVFP4 backend selection, correctness, and performance.
2. SGLang RTX6000 Pro Blackwell FP8 attention/backend/shared-memory behavior.

The project must produce maintainable code, minimal repros, benchmark evidence, and upstream-ready PR/issue artifacts.

## Core principle

No claim without evidence.

Never say a bug is fixed, a backend is selected, performance improved, or output is correct unless there is saved evidence under `results/`, `repros/`, or `upstream/`.

## Mandatory GPU contention policy

The machine may be shared. Other people or processes may use the GPUs.

Every command that could allocate GPU memory or run a GPU kernel must use the GPU lock wrapper:

```bash
python scripts/run_with_gpu_lock.py --gpus 0 --min-free-gb 70 -- <command>
python scripts/run_with_gpu_lock.py --gpus 0,1 --min-free-gb 70 -- <command>
```

Before running any benchmark or server:

1. Run `python scripts/gpu_guard.py status`.
2. Save status logs under `results/gpu_status/` or let the wrapper do it.
3. Use `CUDA_VISIBLE_DEVICES` through the wrapper.
4. Abort or wait if unrelated processes occupy the target GPUs.
5. Mark any run with active unrelated GPU processes as `contended=true` and exclude it from headline metrics.

Do not kill other users' processes. Do not assume idle GPUs. Do not use both GPUs for exploratory work if one GPU is enough.

## Agent orchestration

Use subagents only when it helps:

- explorer agents: read-only code mapping and issue/doc triage.
- patch engineers: focused implementation.
- benchmark engineer: scripts, metrics, result schema.
- reviewer: correctness, maintainability, and unsupported-claim checks.
- hardware sentinel: GPU contention, environment, run validity.

Do not spawn GPU-running subagents concurrently. GPU jobs must be serialized unless the benchmark explicitly tests concurrency and holds the relevant locks.

## Expected workflow

1. Read `MASTER_GOAL.md`, `PROJECT_BRIEF.md`, `TASK_BOARD.md`, and `docs/gpu_contention_policy.md`.
2. Map current repo state.
3. Use explorer subagents to map vLLM/SGLang codepaths.
4. Implement or refine local harness code.
5. Run CPU-only tests first.
6. Run GPU smoke tests with the lock wrapper.
7. Run target repros with the lock wrapper.
8. Save raw logs and environment metadata.
9. Update `TASK_BOARD.md`.
10. Create or update upstream PR/issue drafts.
11. Run reviewer subagent before declaring completion.

## Repo layout

- `scripts/` — local utilities, GPU guard, environment verifier, goal loop.
- `benchmarks/` — serving benchmark client and result summarizers.
- `repros/vllm_mxfp4_sm120/` — vLLM repro scripts and logs.
- `repros/sglang_attention_backend_sm120/` — SGLang repro scripts and logs.
- `docs/` — technical notes and protocols.
- `upstream/` — PR summaries, issue comments, patch notes.
- `.codex/agents/` — custom Codex subagent definitions.
- `results/` — generated artifacts only; do not commit huge files unless explicitly needed.

## Engineering standards

- Prefer small, reviewable patches.
- Add tests before or with changes.
- Scripts must have `--help` and clear failure messages.
- Benchmark scripts must save JSONL raw data.
- Summaries must include command, commit, model, dtype, backend, GPU IDs, contention status, and timestamp.
- If a run fails, save the failure log and classify the failure.
- If a model is gated/unavailable, document exact blocker and choose a smaller accessible smoke model.

## Upstream patch standards

A patch candidate is not upstream-ready unless it has:

- minimal reproducer,
- before/after behavior,
- correctness check,
- evidence of no obvious regression,
- environment metadata,
- and a maintainer-readable summary.

Do not widen hardware capability checks without explaining why the target hardware supports the chosen kernel path. If SM120 differs from SM100, prefer explicit detection, guarded fallback, or clear error messages.

## Benchmark standards

Every benchmark must record:

- framework and commit,
- model ID/path,
- dtype / quantization,
- backend flags,
- tensor parallel size,
- GPU IDs and CUDA_VISIBLE_DEVICES,
- driver / CUDA / PyTorch / Triton versions,
- input length,
- output length,
- concurrency,
- warmup count,
- trial count,
- TTFT,
- TPOT,
- output tokens/sec,
- request latency P50/P95/P99 where applicable,
- GPU memory before/after,
- GPU contention status.

## Done means

The task is done only when:

- relevant tests pass,
- generated artifacts are saved,
- `TASK_BOARD.md` is updated,
- claims are supported by evidence,
- and a reviewer pass has no blocking issues.

## Safety and operational constraints

- Do not run destructive shell commands.
- Do not remove logs or benchmark results unless moving them into an archive with explanation.
- Do not push to upstream remotes without explicit human approval.
- Do not submit PRs automatically; prepare draft text and wait for human review.
- Do not install system packages without explaining why.
- Use isolated venv/conda/docker environments where feasible.
