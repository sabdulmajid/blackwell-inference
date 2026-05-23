# Final Review Gate

Status: not ready for public performance claims or upstream PR submission.

## Blocking Issues

1. No real serving benchmark exists yet. `results/benchmarks/summary.md` contains
   only dry-run rows and all rows are `headline_eligible=False`.
2. Current GPU status blocks real benchmark launches under the required
   `--min-free-gb 70` policy. GPU 0 and GPU 1 both had active `python` PID
   `2999453` and less than 70 GiB free during the benchmark-matrix pass.
3. vLLM has static selector evidence only. It still lacks locked runtime logs,
   selected backend logs, correctness output, and target repro evidence.
4. SGLang has source mapping and local SM120 device evidence only. It lacks an
   installed runtime, target repro logs, backend selection logs, and correctness
   output.
5. The repo has no initial commit, so environment metadata records
   `git_commit=null`.

## Non-Blocking Issues

- Future benchmark runs need exact token usage metadata. The benchmark client now
  requests streamed usage metadata by default.
- Future summaries need GPU before/after memory. The summarizer now reads wrapper
  `gpu_before.json` and `gpu_after.json` when present.
- Upstream drafts are useful internal issue-update drafts, not PR-ready text.

## Readiness

- Resume-ready as final results: no.
- Resume-ready as a rigorous scaffold/investigation project: partly, if described
  honestly as a safety-first harness with mapped codepaths and blocked runtime
  benchmarks.
- Upstream PR ready: no.
- Upstream issue update ready: not yet; both drafts need at least one locked
  runtime repro or a documented install/model-access blocker that maintainers
  can act on.
