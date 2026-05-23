I want you to start the SM120 Blackwell inference-backend project as the master agent.

Before changing anything:

1. Read `AGENTS.md`, `PROJECT_BRIEF.md`, `MASTER_GOAL.md`, `TASK_BOARD.md`, and `docs/gpu_contention_policy.md`.
2. Inspect the repo layout.
3. Do not run GPU-heavy commands yet.
4. Spawn read-only subagents only:
   - `hardware_sentinel` to review the GPU contention policy and status commands.
   - `vllm_explorer` to plan how to map vLLM backend-selection codepaths.
   - `sglang_explorer` to plan how to map SGLang attention/backend codepaths.
   - `benchmark_engineer` to review the benchmark harness requirements.
5. Wait for all subagents and consolidate their findings.

Then produce a concrete plan for the first implementation iteration.

Constraints:

- No GPU-heavy commands on this first run.
- No framework installs unless you explain exactly why.
- Do not edit upstream checkouts yet.
- Update `TASK_BOARD.md` only for tasks you actually complete.
- If you edit files, make small edits only to improve setup clarity or tests.

Done when:

- You have a prioritized plan for vLLM repro, SGLang repro, and benchmark harness.
- You list exact next commands to run.
- You identify which tasks require human review or model access.
