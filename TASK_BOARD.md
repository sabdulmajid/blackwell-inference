# Task Board: blackwell-inference

Codex must update this file after each goal-loop iteration.

Legend:

- `TODO`: not started.
- `DOING`: active.
- `BLOCKED`: blocked with evidence.
- `DONE`: acceptance criteria met.

## 0. Project setup

| Task | Status | Evidence / notes |
|---|---:|---|
| Install Python deps | TODO | Existing environment is sufficient for first-pass tests; no package install was performed. |
| Verify Codex can read AGENTS.md | DONE | First-pass master agent read `AGENTS.md`, `PROJECT_BRIEF.md`, `MASTER_GOAL.md`, `TASK_BOARD.md`, docs, prompts, scripts, repros, and upstream drafts. |
| Record missing `.codex/` config | DONE | `.codex/config.toml` and `.codex/agents/*.toml` are absent in this checkout; not required for harness execution. |
| Create initial git checkpoint | DONE | Public scaffold committed and pushed to `origin/main` at `c55174365f7ce689418f2dbf77849657d76c7470`. |
| Run `verify_blackwell.py` | DONE | Latest controlled smoke artifact: `results/env/verify_blackwell.json`. Default mode is metadata-only; PyTorch CUDA probing is skipped unless `--probe-cuda` runs under the GPU lock wrapper. |
| Run `gpu_guard.py status` | DONE | Latest meaningful-goal artifact: `results/gpu_status/meaningful_goal_final_status.json`; both GPUs had another user's active Python process PID `507867`, so no valid GPU runtime repro was launched. |
| Confirm GPU lock wrapper works with `sleep 5` | DONE | `results/gpu_runs/20260520T091452Z_first_pass_lock_sleep/run_meta.json`; non-GPU sleep command completed and was marked `invalid_contended` because GPU 0 had an existing process. |

## 1. vLLM SM120 FP4/MXFP4/NVFP4

| Task | Status | Evidence / notes |
|---|---:|---|
| Clone/check out vLLM | DONE | Sparse checkout exists at `external/vllm`, clean detached HEAD `5bb8d2767a2829b56e58c68fa8f380e9e4e2bd3e`. `external/` remains ignored and is not committed. |
| Record vLLM commit | DONE | Current inspected source commit: `5bb8d2767a2829b56e58c68fa8f380e9e4e2bd3e`. Installed wheel is vLLM `0.12.0` and may differ materially from source HEAD. |
| Map backend-selection codepaths | DONE | `docs/vllm_sm120_backend_selection.md` and `docs/current_upstream_source_audit.md`; current source map includes `oracle/nvfp4.py`, `oracle/mxfp4.py`, `experts/flashinfer_b12x_moe.py`, `experts/trtllm_mxfp4_moe.py`, `flashinfer_utils.py`, `marlin_utils.py`, and `tests/kernels/moe/test_flashinfer_b12x_moe.py`. |
| Write minimal repro script | DOING | `repros/vllm_mxfp4_sm120/run_repro.sh` is parameterized, refuses unlocked GPU launches, supports `--quantization` and `--dry-run`, captures env metadata, and now writes `backend_summary.json` from backend logs. Dry-run command: `results/repros/vllm_mxfp4_sm120/20260520T100045Z/command.txt`. |
| Run small smoke model | BLOCKED | No vLLM GPU run launched. Latest meaningful-goal checks show another user's active PID `507867` on both GPUs: `results/gpu_status/meaningful_goal_start_status.json`, `results/gpu_status/meaningful_goal_gpu0_check.json`, `results/gpu_status/meaningful_goal_gpu1_check.json`. |
| Run target low-precision repro | TODO |  |
| Capture selected backend logs | DOING | Future real repros will write `backend_grep.txt` and `backend_summary.json`. Static selector evidence saved at `results/repros/vllm_mxfp4_sm120/static_backend_selection_probe.json`. |
| Add correctness check | TODO |  |
| Draft or implement patch | DOING | Local diagnostics patch added (`scripts/vllm_mxfp4_static_probe.py`, `scripts/extract_vllm_backend_evidence.py`). No external vLLM source patch is safe without locked SM120 runtime evidence. |
| Run before/after benchmark | TODO |  |
| Draft upstream PR/issue update | DOING | `upstream/vllm/issue_or_pr_draft.md` updated as diagnostics-first issue-update draft. Not PR-ready until runtime evidence is captured. |

## 2. SGLang RTX6000 Pro FP8 attention/shared-memory

| Task | Status | Evidence / notes |
|---|---:|---|
| Clone/check out SGLang | DONE | Sparse checkout exists at `external/sglang`, clean detached HEAD `a5a64a311a39b153d1e4d3d6bcb67e77cdc9aeae`. `sglang`/`sgl-kernel` are still not installed in the current Python environment. |
| Record SGLang commit | DONE | Current inspected source commit: `a5a64a311a39b153d1e4d3d6bcb67e77cdc9aeae`. |
| Map attention/backend-selection codepaths | DONE | `docs/sglang_sm120_attention_backend.md` and `docs/current_upstream_source_audit.md`; current source map covers SM120 capability helpers, Triton extend-attention block sizing, attention defaulting, GDN linear-attention dispatch, FlashInfer GDN decode-only behavior, and FP8 GEMM auto fallback. |
| Write minimal repro script | DONE | `repros/sglang_attention_backend_sm120/run_repro.sh` is parameterized, refuses unlocked GPU launches, supports `--dry-run`, captures `verify_blackwell.py --probe-cuda` under the wrapper, exposes attention/linear-attention/FP8 backend flags, writes `backend_summary.json`, and uses 2 warmups / 5 requests for smoke. Dry-run artifacts: `results/repros/sglang_attention_backend_sm120/sglang_triton_dry_run_20260520T101900Z/command.txt` and `results/repros/sglang_attention_backend_sm120/sglang_flashinfer_decode_dry_run_20260520T101901Z/command.txt`. |
| Run small smoke model | BLOCKED | No installed SGLang runtime and no small local Qwen3-Next/GDN fixture found in this pass. Next step is bootstrap/install or identify a local small model that exercises the GDN path. |
| Run target FP8 repro | TODO | Target Qwen3-Next FP8 model was not run; static/small-model paths must be exhausted first. |
| Capture shared-memory/backend logs | DOING | `scripts/extract_sglang_backend_evidence.py` added and tested; future real runs will write `backend_grep.txt` and `backend_summary.json`. Local SM120 CUDA property artifact: `results/gpu_runs/20260520T100750Z_sglang_device_probe/verify_blackwell_probe_cuda.json`. |
| Add correctness check | TODO |  |
| Draft or implement patch | DOING | Patch plan documented in `docs/sglang_sm120_attention_backend.md`; no external SGLang source patch is safe until a runtime repro identifies the failing path. |
| Run before/after benchmark | TODO |  |
| Draft upstream PR/issue update | DOING | `upstream/sglang/issue_or_pr_draft.md` updated with environment evidence, codepath map, repro commands, suspected root cause, patch direction, risks, and validation requirements. |

## 3. Benchmark harness

| Task | Status | Evidence / notes |
|---|---:|---|
| Implement OpenAI-compatible benchmark client | DONE | `benchmarks/serve_bench.py` supports real streaming and `--dry-run`, refuses unlocked real requests, checks NaN/Inf text, requests streamed usage metadata by default, and records benchmark metadata. |
| Save JSONL raw results | DOING | Benchmark matrix dry-run JSONL saved at `results/benchmarks/20260520T103000Z_stage_a_vllm_tiny_dry_run.jsonl` and `results/benchmarks/20260520T103001Z_stage_a_sglang_tiny_dry_run.jsonl`; real serving benchmark still pending. |
| Save summary CSV/Markdown | DOING | `benchmarks/summarize_results.py` wrote `results/benchmarks/summary.csv` and `results/benchmarks/summary.md` from dry-run summaries; all current rows are `headline_eligible=False`. |
| Add environment metadata | DOING | Benchmark metadata records command, repo commit if available, framework, model, dtype/quantization/backend, GPU lock env, and optional verifier metadata. |
| Add contention detection | DOING | `scripts/run_with_gpu_lock.py` writes before/after GPU snapshots, target process lists, `contention_label`, and `contended`. It supports `--max-wait-seconds`, records `wait_timeout` metadata for blocked smoke attempts, and now gives non-wait `not_eligible` records the same validity fields. Benchmark summaries mark dry runs/non-wrapper runs as non-headline evidence. |
| Add validation checks | DONE | Added argument validation, zero-token invalidation, first-content TTFT, usage-token parsing, dry-run status, raw/summary cross-checking, GPU before/after summarization from wrapper metadata, low-sample validation flags, and `headline_eligible` summary output. |
| Run 1-GPU benchmark | BLOCKED | Real serving benchmarks were not launched because latest checks showed another user's active PID `507867` on both GPUs. A bounded locked CUDA smoke waited 906.5 seconds and timed out without running: `results/gpu_runs/20260523T153219Z_meaningful_goal_torch_cuda_smoke/run_meta.json`. |
| Run 2-GPU benchmark | TODO |  |

## 4. Final artifacts

| Task | Status | Evidence / notes |
|---|---:|---|
| Write vLLM technical note | DONE | `docs/vllm_sm120_backend_selection.md` maps installed and current-upstream codepaths, root-cause hypothesis, safe patch plan, and repro command. |
| Write SGLang technical note | DONE | `docs/sglang_sm120_attention_backend.md` maps current upstream codepaths, local SM120 device properties, root-cause hypothesis, repro commands, and safe patch plan. |
| Write final resume bullets | DONE | `docs/resume_bullets.md`; bullets avoid performance/correctness claims and are tied to existing harness/source-mapping evidence. |
| Run final review subagent pass | DONE | `docs/final_review.md`; reviewer found no unsafe GPU practice or speedup/correctness overclaim, but blocked public/upstream readiness on missing real runtime evidence. |
| Confirm no unsupported claims | DONE | Current docs and upstream drafts state dry runs are not benchmark evidence and that vLLM/SGLang patches are not PR-ready without locked runtime repros. |
| Write evidence ledger | DONE | `docs/evidence_ledger.md` records artifact validity, claims supported/not supported, current blockers, and audit-time GPU status. |
| Run post-queue tests | DONE | `results/tests/post_queue_audit_pytest.txt`; `26 passed in 7.22s`. |

## 5. Upstream patch discipline

| Task | Status | Evidence / notes |
|---|---:|---|
| Record external setup commands | DONE | `docs/external_repo_setup.md`; updated with sparse checkout commands and current inspected commits for vLLM `5bb8d2767a2829b56e58c68fa8f380e9e4e2bd3e` and SGLang `a5a64a311a39b153d1e4d3d6bcb67e77cdc9aeae`. |
| Create vLLM patch plan | DONE | `upstream/vllm/patch_plan.md`; current-source diagnostics/test-first plan only. External vLLM checkout is clean and has no diff. |
| Create SGLang patch plan | DONE | `upstream/sglang/patch_plan.md`; current-source diagnostics/test-first plan only. External SGLang checkout is clean and has no diff. |
| Summarize external diffs | DONE | `upstream/vllm/diff_summary.md` and `upstream/sglang/diff_summary.md` record no external source diffs and define the intended small diff shape. |
| Add upstream no-silly-mistakes checklist | DONE | `docs/upstream_pr_protocol.md`; checklist rejects broad hardware assumptions, stale SHAs, large-model-only tests, unsupported performance claims, dirty checkouts, and uncontrolled backend forcing. |
| Create external vLLM patch branch | TODO | `external/vllm` is clean at the inspected commit. Branch creation is intentionally deferred until a diagnostics patch is written. |
| Create external SGLang patch branch | TODO | `external/sglang` is clean at the inspected commit. Branch creation is intentionally deferred until a diagnostics patch is written. |
| Run patch-discipline tests | DONE | `results/tests/upstream_patch_discipline_pytest.txt`; `26 passed in 7.12s`. No external tests were run in that pass; current external checkouts are source-inspection only. |

## 6. Reproducibility and dependencies

| Task | Status | Evidence / notes |
|---|---:|---|
| Write reproducibility guide | DONE | `docs/reproducibility.md`; includes fresh clone `.venv` setup, dry-run validation, locked one-GPU smoke command shape, target repro command shapes, GPU contention policy, archive commands, and schema validation. |
| Write dependency matrix | DONE | `docs/dependency_matrix.md`; current observed versions are also saved in `results/env/versions.json`. |
| Write model access plan | DONE | `docs/model_access_plan.md`; separates tiny smoke, small meaningful, vLLM target, and SGLang target models and marks large/gated download risks. |
| Add version/repro scripts | DONE | `scripts/collect_versions.py`, `scripts/check_reproducibility.py`, and `scripts/archive_results.py`; all are lightweight and avoid CUDA initialization. |
| Run reproducibility check | DONE | `results/env/reproducibility_check.json`; status `ok` with documented warnings for missing optional/dev packages, non-repo current venv, absent repo-local `.venv`, and optional framework gaps. |
| Run reproducibility tests | DONE | `results/tests/reproducibility_pytest.txt`; `29 passed, 1 skipped in 0.77s`. The skipped test is an optional vLLM import/static-probe test gated by `BLACKWELL_INFERENCE_RUN_OPTIONAL_IMPORT_TESTS=1`. |
| Run impact source-audit tests | DONE | `results/tests/impact_source_audit_pytest.txt`; `29 passed, 1 skipped in 0.77s`. |
| Run meaningful-goal tests | DONE | `results/tests/meaningful_goal_pytest.txt`; lightweight tests pass after documenting the blocked GPU smoke. |
