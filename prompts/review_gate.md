Run a final review gate for the current branch.

Read:

- `AGENTS.md`
- `MASTER_GOAL.md`
- `TASK_BOARD.md`
- `docs/benchmark_protocol.md`
- `docs/upstream_pr_protocol.md`
- all files under `upstream/`
- latest result summaries under `results/`

Spawn reviewer and hardware_sentinel subagents.

Check for:

1. unsupported claims,
2. missing environment metadata,
3. GPU contention violations,
4. benchmark methodology flaws,
5. overbroad SM100/SM120 assumptions,
6. missing correctness checks,
7. weak PR/issue drafts,
8. stale task board status.

Do not edit files until after the subagents report. Then make only documentation/status fixes that are clearly justified.

Return:

- blocking issues,
- non-blocking issues,
- whether the repo is ready to show on a resume,
- whether any upstream PR/issue draft is ready for human submission.
