# Evidence Ledger

Audit timestamp: `2026-05-20T10:31:55Z`

Latest source-audit update: `2026-05-23T15:04Z`

Scope: post-queue audit after first-pass setup, controlled smoke, real-GPU smoke
attempt, vLLM deep dive, SGLang deep dive, and benchmark-matrix ramp-up. No
new GPU-heavy workload, model download, server launch, or two-GPU command was
run during this audit.

## 2026-05-23 Source-Audit Update

This update cloned sparse, ignored external source trees for current-upstream
inspection only. It did not run GPU kernels, download models, start servers, or
modify upstream source.

| Artifact | Type | Command / source | Framework | GPU IDs | Validity | Supports | Does not support |
|---|---|---|---|---|---|---|---|
| `docs/current_upstream_source_audit.md` | doc/source-audit | sparse source inspection under `external/vllm` and `external/sglang` | vLLM/SGLang | none | valid source evidence | Current upstream codepath map and revised patch direction | Runtime backend selection, correctness, performance |
| `external/vllm` | external-source | sparse clone of `https://github.com/vllm-project/vllm.git` | vLLM | none | clean local source input | Source inspected at `5bb8d2767a2829b56e58c68fa8f380e9e4e2bd3e` | Any local patch or runtime behavior |
| `external/sglang` | external-source | sparse clone of `https://github.com/sgl-project/sglang.git` | SGLang | none | clean local source input | Source inspected at `a5a64a311a39b153d1e4d3d6bcb67e77cdc9aeae` | Any local patch or runtime behavior |
| `results/gpu_status/impact_resume_status.json` | gpu-status | `python scripts/gpu_guard.py status --out results/gpu_status/impact_resume_status.json` | harness | none | contended status evidence | Active Python work was present on both GPUs, so no GPU smoke was attempted | Permission to run a benchmark |

New source-supported claims:

- vLLM current `main` has an explicit SM12x NVFP4 FlashInfer B12x MoE expert,
  an SM120-gated kernel test, and an explicit `flashinfer_b12x` backend option.
- SGLang current `main` has explicit SM120 capability helpers, SM120 Triton
  extend-attention block sizing for smaller workstation shared-memory limits,
  and an SM120 FP8 GEMM auto fallback to Triton.

Still not proven:

- any real vLLM or SGLang backend selected on this machine;
- any correctness, latency, throughput, speedup, or regression;
- whether current upstream already resolves the original runtime issue.

## Required Audit Commands

| Command | Result | Evidence |
|---|---|---|
| `pwd` | repo root (`blackwell-inference`) | terminal output |
| `git status --short` | dirty/untracked working tree; all project files are untracked | terminal output |
| `git log --oneline -5` | historical audit result: failed because branch `main` had no commits at that time | terminal output |
| `find results -maxdepth 4 -type f \| sort \| sed -n '1,200p'` | listed current result artifacts | terminal output |
| `python scripts/gpu_guard.py status --out results/gpu_status/post_queue_audit_status.json` | completed; both GPUs had active PID `2999453` | `results/gpu_status/post_queue_audit_status.json` |
| `pytest -q` | passed: `26 passed in 7.22s` | `results/tests/post_queue_audit_pytest.txt` |

## Current Repository State

- Current pushed baseline commit SHA: `c55174365f7ce689418f2dbf77849657d76c7470`.
- Working tree after the source-audit pass contains documentation updates that
  should be committed after tests pass.
- Framework packages observed in environment metadata:
  - vLLM: `0.12.0`
  - SGLang: not installed
  - PyTorch: `2.9.0`
  - Triton: `3.5.0`
  - FlashInfer: `0.5.3`
- External source trees:
  - `external/vllm`: clean sparse checkout at
    `5bb8d2767a2829b56e58c68fa8f380e9e4e2bd3e`
  - `external/sglang`: clean sparse checkout at
    `a5a64a311a39b153d1e4d3d6bcb67e77cdc9aeae`

## Artifact Ledger

| Path | Type | Producing command | Timestamp | Framework | GPU IDs | Validity | Supports | Does not support |
|---|---|---|---|---|---|---|---|---|
| `results/env/verify_blackwell.json` | env | `python scripts/verify_blackwell.py --out results/env/verify_blackwell.json` | `2026-05-20T09:27:46Z` | harness | none | valid metadata-only | Environment capture works without CUDA allocation | CUDA compute capability, runtime backend, performance |
| `results/env/pre_smoke_verify_blackwell.json` | env | `python scripts/verify_blackwell.py --out results/env/pre_smoke_verify_blackwell.json` | `2026-05-20T09:31:15Z` | harness | none | valid metadata-only | Pre-smoke environment snapshot | CUDA probe/runtime behavior |
| `results/env/sglang_deep_dive_verify_blackwell.json` | env | `python scripts/verify_blackwell.py --out results/env/sglang_deep_dive_verify_blackwell.json` | `2026-05-20T10:20:37Z` | SGLang/harness | none | valid metadata-only | SGLang absent, package versions captured | SGLang runtime behavior |
| `results/env/benchmark_matrix_verify_blackwell.json` | env | `python scripts/verify_blackwell.py --out results/env/benchmark_matrix_verify_blackwell.json` | `2026-05-20T10:23:18Z` | harness | none | valid metadata-only | Benchmark-matrix environment and package versions | CUDA allocation/backend performance |
| `results/env/versions.json` | env | `python scripts/collect_versions.py --out results/env/versions.json` | current | harness | none | valid sanitized metadata | Current package versions and redacted environment state without CUDA initialization | Runtime backend behavior |
| `results/env/reproducibility_check.json` | env/test | `python scripts/check_reproducibility.py --out results/env/reproducibility_check.json` | current | harness | none | valid check with warnings | Reproducibility prerequisites and known gaps are machine-readable | Framework install success or GPU readiness |
| `results/gpu_status/sglang_device_probe_pre_status.json` | gpu-status | `python scripts/gpu_guard.py status --out ...` | `2026-05-20T10:07:42Z` | harness | none | valid | Both GPUs idle before locked CUDA device probe | Benchmark performance |
| `results/gpu_status/sglang_device_probe_gpu0_check.json` | gpu-status | `python scripts/gpu_guard.py check --gpus 0 --min-free-gb 70 --out ...` | `2026-05-20T10:07:42Z` | harness | GPU 0 | valid | GPU 0 was eligible for the CUDA property probe | Serving benchmark validity |
| `results/gpu_runs/20260520T100750Z_sglang_device_probe/run_meta.json` | gpu-run | `python scripts/run_with_gpu_lock.py --gpus 0 --min-free-gb 70 --wait --max-wait-seconds 300 --label sglang_device_probe -- bash -lc 'python scripts/verify_blackwell.py --probe-cuda --out "$BLACKWELL_INFERENCE_GPU_RUN_DIR/verify_blackwell_probe_cuda.json"'` | `2026-05-20T10:07:50Z` | harness/SGLang | physical GPU 0 | valid | Lock wrapper works for a CUDA metadata probe; run marked `valid_uncontended` | Model serving, backend selection, correctness, throughput |
| `results/gpu_runs/20260520T100750Z_sglang_device_probe/verify_blackwell_probe_cuda.json` | env/gpu-run | same as above | `2026-05-20T10:07:51Z` | harness/SGLang | visible logical 0 mapped to physical GPU 0 | valid | PyTorch sees RTX PRO 6000 Blackwell, capability `(12, 0)`, total memory `101971591168`, `shared_memory_per_multiprocessor=102400` | SGLang server behavior, Triton requested shared memory |
| `results/gpu_runs/20260520T091452Z_first_pass_lock_sleep/run_meta.json` | gpu-run | `python scripts/run_with_gpu_lock.py ... --label first_pass_lock_sleep -- bash -lc 'echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; sleep 5'` | `2026-05-20T09:14:52Z` | harness | physical GPU 0 | contended | Wrapper logs command/GPU mapping and marks contention | Benchmark validity or GPU compute correctness |
| `results/gpu_runs/20260520T092313Z_first_pass_wrapped_dry_bench/run_meta.json` | gpu-run | wrapper around `benchmarks/serve_bench.py --dry-run` | `2026-05-20T09:23:13Z` | harness | physical GPU 0 | contended/dry-run | Wrapper can capture a dry-run benchmark directory | Real benchmark performance |
| `results/gpu_runs/20260520T092313Z_first_pass_wrapped_dry_bench/bench_dry.jsonl` | benchmark | wrapped `benchmarks/serve_bench.py --framework other --model dry-run-model --dry-run` | `2026-05-20T09:23:13Z` | harness | physical GPU 0 env only | dry-run/contended wrapper | Benchmark JSONL schema | Server behavior, performance |
| `results/gpu_runs/20260520T092313Z_first_pass_wrapped_dry_bench/bench_dry.summary.json` | benchmark | same as above | `2026-05-20T09:23:13Z` | harness | physical GPU 0 env only | dry-run/contended wrapper | Benchmark summary schema | Headline metrics |
| `results/gpu_runs/20260520T093325Z_torch_cuda_smoke_gpu1/gpu_before.json` | gpu-run | early wrapper attempt for `scripts/torch_cuda_smoke.py` | `2026-05-20T09:33:25Z` | harness | physical GPU 1 | failed/incomplete | Attempt began with GPU status capture | CUDA smoke success |
| `results/gpu_runs/20260520T094432Z_torch_cuda_smoke_gpu1_waitcheck/run_meta.json` | gpu-run | wrapper attempt for `scripts/torch_cuda_smoke.py --size 256 --dtype float16` | `2026-05-20T09:44:32Z` | harness | physical GPU 1 | failed/contended | GPU lock wrapper recorded `wait_timeout`/`invalid_contended`; command did not run | Successful CUDA smoke |
| `results/gpu_status/pre_real_smoke_status.json` | gpu-status | `python scripts/gpu_guard.py status --out ...` | `2026-05-20T09:31:15Z` | harness | none | contended | Active PID `2805853` occupied both GPUs | Valid benchmark window |
| `results/gpu_status/pre_real_smoke_gpu0_check.json` | gpu-status | `python scripts/gpu_guard.py check --gpus 0 --min-free-gb 70 --out ...` | `2026-05-20T09:31:15Z` | harness | GPU 0 | contended | GPU 0 blocked by active process | Valid GPU run |
| `results/gpu_status/pre_real_smoke_gpu1_check.json` | gpu-status | `python scripts/gpu_guard.py check --gpus 1 --min-free-gb 70 --out ...` | `2026-05-20T09:31:15Z` | harness | GPU 1 | contended | GPU 1 blocked by active process | Valid GPU run |
| `results/gpu_status/vllm_deep_dive_status_latest.json` | gpu-status | `python scripts/gpu_guard.py status --out ...` | `2026-05-20T09:54:23Z` | vLLM/harness | none | contended | Active PID `2805853` during vLLM deep dive | vLLM runtime evidence |
| `results/gpu_status/vllm_deep_dive_gpu0_check.json` | gpu-status | `python scripts/gpu_guard.py check --gpus 0 --min-free-gb 70 --out ...` | `2026-05-20T09:48:37Z` | vLLM/harness | GPU 0 | contended | GPU 0 blocked for vLLM repro | vLLM backend selection runtime |
| `results/gpu_status/vllm_deep_dive_gpu1_check.json` | gpu-status | `python scripts/gpu_guard.py check --gpus 1 --min-free-gb 70 --out ...` | `2026-05-20T09:48:37Z` | vLLM/harness | GPU 1 | contended | GPU 1 blocked for vLLM repro | vLLM backend selection runtime |
| `results/gpu_status/vllm_deep_dive_final_status.json` | gpu-status | `python scripts/gpu_guard.py status --out ...` | `2026-05-20T10:04:14Z` | vLLM/harness | none | valid status only | Later idle window was observed | No vLLM run occurred in that window |
| `results/gpu_status/sglang_deep_dive_status.json` | gpu-status | `python scripts/gpu_guard.py status --out ...` | `2026-05-20T10:06:10Z` | SGLang/harness | none | valid status only | Idle window before SGLang CUDA property probe | No SGLang server run |
| `results/gpu_status/sglang_deep_dive_final_status_latest.json` | gpu-status | `python scripts/gpu_guard.py status --out ...` | `2026-05-20T10:21:13Z` | SGLang/harness | none | contended | Later active PID `2999453` on both GPUs | Valid benchmark window |
| `results/gpu_status/benchmark_matrix_pre_status.json` | gpu-status | `python scripts/gpu_guard.py status --out ...` | `2026-05-20T10:23:18Z` | harness | none | contended | Active PID `2999453`, GPU 0 ~63.25 GiB free, GPU 1 ~67.07 GiB free | Real benchmark eligibility |
| `results/gpu_status/benchmark_matrix_gpu0_check.json` | gpu-status | `python scripts/gpu_guard.py check --gpus 0 --min-free-gb 70 --out ...` | `2026-05-20T10:23:31Z` | harness | GPU 0 | contended/invalid | GPU 0 below 70 GiB threshold | Real benchmark eligibility |
| `results/gpu_status/benchmark_matrix_gpu1_check.json` | gpu-status | `python scripts/gpu_guard.py check --gpus 1 --min-free-gb 70 --out ...` | `2026-05-20T10:23:31Z` | harness | GPU 1 | contended/invalid | GPU 1 below 70 GiB threshold | Real benchmark eligibility |
| `results/gpu_status/benchmark_matrix_final_status.json` | gpu-status | `python scripts/gpu_guard.py status --out ...` | `2026-05-20T10:29:42Z` | harness | none | contended | Active PID `2999453` persisted | Real benchmark eligibility |
| `results/gpu_status/benchmark_matrix_final_gpu1_check.json` | gpu-status | `python scripts/gpu_guard.py check --gpus 1 --min-free-gb 70 --out ...` | `2026-05-20T10:30:22Z` | harness | GPU 1 | contended/invalid | GPU 1 still below 70 GiB threshold | Real benchmark eligibility |
| `results/gpu_status/post_queue_audit_status.json` | gpu-status | `python scripts/gpu_guard.py status --out results/gpu_status/post_queue_audit_status.json` | `2026-05-20T10:31:55Z` | harness | none | contended | Current audit-time status: active PID `2999453`, GPU 0 ~60.42 GiB free, GPU 1 ~66.88 GiB free | Permission to run real benchmark |
| `results/benchmarks/20260520T103000Z_stage_a_vllm_tiny_dry_run.jsonl` | benchmark | `python benchmarks/serve_bench.py --dry-run --framework vllm --model hf-internal-testing/tiny-random-gpt2 ...` | `2026-05-20T10:27:00Z` | vLLM/harness | none | dry-run | Benchmark client raw JSONL schema for vLLM Stage A | vLLM server, latency, throughput |
| `results/benchmarks/20260520T103000Z_stage_a_vllm_tiny_dry_run.summary.json` | benchmark | same as above | `2026-05-20T10:27:00Z` | vLLM/harness | none | dry-run | Summary schema; `include_usage=true`; dry-run validity label | Headline benchmark numbers |
| `results/benchmarks/20260520T103001Z_stage_a_sglang_tiny_dry_run.jsonl` | benchmark | `python benchmarks/serve_bench.py --dry-run --framework sglang --model hf-internal-testing/tiny-random-gpt2 ...` | `2026-05-20T10:27:00Z` | SGLang/harness | none | dry-run | Benchmark client raw JSONL schema for SGLang Stage A | SGLang server, latency, throughput |
| `results/benchmarks/20260520T103001Z_stage_a_sglang_tiny_dry_run.summary.json` | benchmark | same as above | `2026-05-20T10:27:00Z` | SGLang/harness | none | dry-run | Summary schema; `include_usage=true`; dry-run validity label | Headline benchmark numbers |
| `results/benchmarks/20260520T102402Z_stage_b_vllm_meaningful_dry_run.summary.json` | benchmark | `benchmarks/serve_bench.py --dry-run --framework vllm --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ...` | `2026-05-20T10:24:26Z` | vLLM/harness | none | dry-run | Stage B planned schema | Any model performance |
| `results/benchmarks/20260520T102403Z_stage_b_sglang_meaningful_dry_run.summary.json` | benchmark | `benchmarks/serve_bench.py --dry-run --framework sglang --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ...` | `2026-05-20T10:24:26Z` | SGLang/harness | none | dry-run | Stage B planned schema | Any model performance |
| `results/benchmarks/summary.csv` | benchmark | `python benchmarks/summarize_results.py --results results/benchmarks ...` | after `2026-05-20T10:27:00Z` | harness | none | dry-run summaries only | Aggregates current summaries and marks all rows `headline_eligible=False` | Any performance claim |
| `results/benchmarks/summary.md` | benchmark | same as above | after `2026-05-20T10:27:00Z` | harness | none | dry-run summaries only | Human-readable benchmark summary with no headline-eligible rows | Any performance claim |
| `results/repros/vllm_mxfp4_sm120/static_backend_selection_probe.json` | repro/static | `python scripts/vllm_mxfp4_static_probe.py --out ...` | `2026-05-20T10:00:29Z` | vLLM | none | valid static evidence | Installed vLLM `0.12.0` selector simulation returns `MARLIN` for mocked SM120 MXFP4 scenarios | Runtime backend selection, correctness, kernel support |
| `results/repros/vllm_mxfp4_sm120/20260520T100045Z/command.txt` | repro | `bash repros/vllm_mxfp4_sm120/run_repro.sh --dry-run ...` | `2026-05-20T10:00:45Z` | vLLM | none | dry-run | vLLM repro command construction | vLLM server behavior |
| `results/repros/vllm_mxfp4_sm120/20260520T100045Z/status.txt` | repro | same as above | `2026-05-20T10:00:45Z` | vLLM | none | dry-run | Repro dry-run status | Runtime success |
| `results/repros/sglang_attention_backend_sm120/sglang_triton_dry_run_20260520T101900Z/command.txt` | repro | `bash repros/sglang_attention_backend_sm120/run_repro.sh --dry-run --attention-backend triton ...` | `2026-05-20T10:19:00Z` | SGLang | none | dry-run | Triton SGLang repro command construction | SGLang runtime failure/success |
| `results/repros/sglang_attention_backend_sm120/sglang_triton_dry_run_20260520T101900Z/status.txt` | repro | same as above | `2026-05-20T10:19:00Z` | SGLang | none | dry-run | Dry-run status | Runtime success |
| `results/repros/sglang_attention_backend_sm120/sglang_flashinfer_decode_dry_run_20260520T101901Z/command.txt` | repro | `bash repros/sglang_attention_backend_sm120/run_repro.sh --dry-run --attention-backend flashinfer --linear-attn-decode-backend flashinfer ...` | `2026-05-20T10:19:01Z` | SGLang | none | dry-run | FlashInfer-decode/Triton-prefill repro command construction | SGLang runtime fallback validity |
| `results/repros/sglang_attention_backend_sm120/sglang_flashinfer_decode_dry_run_20260520T101901Z/status.txt` | repro | same as above | `2026-05-20T10:19:01Z` | SGLang | none | dry-run | Dry-run status | Runtime success |
| `upstream/vllm/issue_or_pr_draft.md` | upstream-draft | documentation edit | current | vLLM | none | partially supported draft | Maintainer-readable issue-update skeleton grounded in static probe/codepath map | PR readiness, runtime behavior |
| `upstream/vllm/patch_plan.md` | upstream-draft | documentation edit | current | vLLM | none | valid plan, not evidence of a fix | Minimal diagnostics/test-first patch plan and external repo status | Runtime backend behavior or PR readiness |
| `upstream/vllm/diff_summary.md` | upstream-draft | documentation edit | current | vLLM | none | valid no-diff summary | Confirms clean external vLLM checkout with no source diff | Patch correctness |
| `upstream/sglang/issue_or_pr_draft.md` | upstream-draft | documentation edit | current | SGLang | none | partially supported draft | Maintainer-readable investigation skeleton grounded in source map and CUDA property probe | PR readiness, runtime behavior |
| `upstream/sglang/patch_plan.md` | upstream-draft | documentation edit | current | SGLang | none | valid plan, not evidence of a fix | Minimal diagnostics/test-first patch plan and external repo status | Runtime backend behavior or PR readiness |
| `upstream/sglang/diff_summary.md` | upstream-draft | documentation edit | current | SGLang | none | valid no-diff summary | Confirms clean external SGLang checkout with no source diff | Patch correctness |
| `docs/external_repo_setup.md` | doc | documentation edit | current | harness | none | valid setup plan | Exact sparse checkout commands and intended patch branches | External source state until commands are run |
| `docs/reproducibility.md` | doc | documentation edit | current | harness | none | valid setup plan | Fresh clone `.venv` setup, dry-run validation, locked smoke command shapes, archive flow | Runtime benchmark validity |
| `docs/dependency_matrix.md` | doc | documentation edit | current | harness | none | valid dependency plan | Base vs optional dependency state and risks | Successful optional framework installs |
| `docs/model_access_plan.md` | doc | documentation edit | current | harness | none | valid access plan | Model categories, download/gating risks, dry-run command shapes | Actual model availability |
| `docs/vllm_sm120_backend_selection.md` | doc | documentation edit | current | vLLM | none | partially supported | vLLM codepath map and static root-cause hypothesis | Runtime fix or performance |
| `docs/sglang_sm120_attention_backend.md` | doc | documentation edit | current | SGLang | GPU 0 for property probe | partially supported | SGLang codepath map and local SM120 device properties | SGLang server repro |
| `docs/benchmark_matrix.md` | doc | documentation edit | current | harness | none | valid plan/status | Staged matrix and current GPU contention blocker | Benchmark results |
| `docs/final_review.md` | doc | documentation edit | current | harness | none | valid review status | Public/upstream readiness blockers | Performance/correctness |
| `docs/resume_bullets.md` | doc | documentation edit | current | harness | none | valid conservative bullets | Evidence-backed scaffold/source-mapping bullets | Speedup/correctness claims |
| `results/tests/post_queue_audit_pytest.txt` | test | `pytest -q` | audit final run | harness | none | valid | Current lightweight test suite passes: `26 passed in 7.22s`; includes wrapper not-eligible metadata coverage | Runtime benchmark validity |
| `results/tests/upstream_patch_discipline_pytest.txt` | test | `pytest -q` | patch-discipline pass | harness | none | valid | Current lightweight test suite passes: `26 passed in 7.12s` after patch-plan/doc updates | Runtime benchmark validity |
| `results/tests/reproducibility_pytest.txt` | test | `pytest -q` | reproducibility pass | harness | none | valid | Current lightweight test suite passes: `29 passed, 1 skipped in 0.77s`; skip is optional vLLM import/static-probe test | Runtime benchmark validity |

## Claims Table

### Proven By Current Evidence

| Claim | Evidence |
|---|---|
| `scripts/gpu_guard.py status` works and records current GPU status. | `results/gpu_status/post_queue_audit_status.json` |
| `scripts/run_with_gpu_lock.py` can set physical GPU IDs and save before/after metadata for a lightweight command. | `results/gpu_runs/20260520T091452Z_first_pass_lock_sleep/run_meta.json` |
| Locked PyTorch CUDA metadata probing works on one GPU when uncontended. | `results/gpu_runs/20260520T100750Z_sglang_device_probe/run_meta.json` and `verify_blackwell_probe_cuda.json` |
| The local GPU seen in the locked CUDA probe is RTX PRO 6000 Blackwell with compute capability `(12, 0)`. | `results/gpu_runs/20260520T100750Z_sglang_device_probe/verify_blackwell_probe_cuda.json` |
| Installed vLLM `0.12.0` static MXFP4 selector simulation returns `MARLIN` for mocked SM120 scenarios. | `results/repros/vllm_mxfp4_sm120/static_backend_selection_probe.json` |
| The benchmark client and summarizer can produce raw JSONL, JSON summaries, CSV, and Markdown dry-run schema artifacts. | `results/benchmarks/20260520T103000Z_stage_a_vllm_tiny_dry_run.jsonl`, `results/benchmarks/20260520T103001Z_stage_a_sglang_tiny_dry_run.jsonl`, `results/benchmarks/summary.csv`, `results/benchmarks/summary.md` |
| Current benchmark summaries do not treat dry runs as headline evidence. | `results/benchmarks/summary.csv`, `results/benchmarks/summary.md` |
| SGLang is not installed in the current Python environment. | `results/env/sglang_deep_dive_verify_blackwell.json`, `results/env/benchmark_matrix_verify_blackwell.json` |
| Current audit-time GPU state is contended and below the 70 GiB threshold on both GPUs. | `results/gpu_status/post_queue_audit_status.json` |
| Current lightweight test suite passes. | `results/tests/post_queue_audit_pytest.txt` |
| Patch-discipline updates did not break local lightweight tests. | `results/tests/upstream_patch_discipline_pytest.txt` |
| Reproducibility state is now documented and checkable without CUDA initialization. | `docs/reproducibility.md`, `docs/dependency_matrix.md`, `docs/model_access_plan.md`, `results/env/versions.json`, `results/env/reproducibility_check.json` |
| Reproducibility scripts pass local tests. | `results/tests/reproducibility_pytest.txt` |

### Partially Supported

| Claim | Evidence | Gap |
|---|---|---|
| vLLM SM120 MXFP4 backend selection likely needs runtime verification against current upstream. | Static selector probe and `docs/vllm_sm120_backend_selection.md` | No current-upstream checkout/runtime repro |
| SGLang Qwen3-Next FP8 failure may involve Triton shared-memory or hybrid GDN linear-attention backend selection. | `docs/sglang_sm120_attention_backend.md`, local device probe | No SGLang runtime logs or stack trace |
| The benchmark matrix is ready to run once GPUs are eligible. | Dry-run artifacts and `docs/benchmark_matrix.md` | No real server benchmark yet |
| Upstream issue drafts are maintainer-shaped. | `upstream/vllm/issue_or_pr_draft.md`, `upstream/sglang/issue_or_pr_draft.md` | Missing runtime/current behavior/correctness evidence |
| Patch plans are reviewable and conservative. | `upstream/vllm/patch_plan.md`, `upstream/sglang/patch_plan.md`, `docs/upstream_pr_protocol.md` | No external checkout or source diff exists yet |

### Not Yet Proven

| Claim | Missing evidence |
|---|---|
| Any vLLM backend actually selected on SM120 for a real FP4/MXFP4/NVFP4 model. | Locked vLLM server logs, `backend_summary.json`, raw benchmark JSONL |
| Any SGLang backend actually selected on SM120 for Qwen3-Next FP8. | Installed SGLang runtime, locked server logs, `backend_summary.json` |
| Any vLLM or SGLang correctness property. | Deterministic prompt/output, NaN/Inf check tied to real run |
| Any tokens/sec, TTFT, TPOT, latency, or throughput metric. | Real raw JSONL and headline-eligible summary |
| Any speedup or regression. | Matched valid uncontended before/after benchmark rows |
| Any two-GPU behavior. | Locked TP=2 run after valid TP=1 baseline |
| Any upstream patch is ready. | Runtime repro, tests, before/after behavior |

### Disproven Or Failed

| Claim/attempt | Evidence | Classification |
|---|---|---|
| The one-GPU PyTorch CUDA smoke ran successfully. | `results/gpu_runs/20260520T094432Z_torch_cuda_smoke_gpu1_waitcheck/run_meta.json` | failed/contended wait timeout; command did not run |
| The benchmark matrix contains real benchmark results. | `results/benchmarks/summary.csv` and `summary.md` | false; all rows are dry runs |
| SGLang runtime repro ran. | SGLang repro directories only contain dry-run `command.txt`/`status.txt`; no server logs | false |
| vLLM runtime repro ran. | vLLM repro directory only contains dry-run `command.txt`/`status.txt` plus static probe | false |

### Blocked

| Blocker | Evidence | Next action |
|---|---|---|
| GPU contention / below threshold | `results/gpu_status/post_queue_audit_status.json`; active PID `2999453`, GPU 0 ~60.42 GiB free, GPU 1 ~66.88 GiB free | Retry guard check later; do not run benchmarks now |
| No runtime source/benchmark evidence at current upstream commits | `docs/current_upstream_source_audit.md` | Run a locked one-GPU smoke after guard eligibility passes |
| SGLang absent | `results/env/benchmark_matrix_verify_blackwell.json` and `sglang_deep_dive_verify_blackwell.json` | Install or checkout SGLang in isolated env |
| External vLLM/SGLang patch branches absent | `TASK_BOARD.md`, local filesystem state | Create branches only when writing small diagnostics patches |
| Target low-precision models not validated locally | no local real model artifact under `results/` | Identify smallest accessible target model or document model-access blocker |

## Safety And Consistency Checks

- No evidence was found of a GPU-heavy model/server benchmark bypassing
  `scripts/run_with_gpu_lock.py`.
- `verify_blackwell.py --probe-cuda` was only used inside the GPU lock wrapper
  in the recorded CUDA property probe.
- Current repro scripts use `kill` only to stop their own launched child server
  PID during cleanup; no command kills unrelated user processes.
- `SM120_LAB_*` names remain only as backward-compatible environment variable
  fallbacks inside scripts/tests. User-facing docs use `blackwell-inference`.
- `scripts/bootstrap_external.sh --help` is non-destructive and exits before any
  clone/fetch operation. Future commit SHA artifacts are written under
  `results/external_commits/`.
- New non-wait `not_eligible` wrapper metadata records validity fields
  (`contention_label`, `contended`, target processes, run directory, and
  selected GPU IDs) consistently with completed and wait-timeout runs.
- External vLLM/SGLang source trees are clean sparse checkouts with no upstream
  source diffs, dirty external branches, or upstream branch result files in this
  workspace.
- All benchmark rows currently in `results/benchmarks/summary.csv` are dry-run
  rows and `headline_eligible=False`.
- No raw JSONL file currently supports a real serving benchmark claim.
- Several result artifacts record `git_commit=null` because the repository has
  no commits yet.

## Audit Conclusion

The repository is a credible harness and investigation scaffold with valid
environment, safety, source-mapping, and dry-run schema evidence. It is not yet
a benchmarked results project and is not ready for upstream patch finalization.
The next phase should focus on getting one uncontended one-GPU real run, then
only expanding after raw logs and correctness evidence exist.
