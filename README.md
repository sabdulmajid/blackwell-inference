# blackwell-inference

`blackwell-inference` is a GPU-safe investigation and benchmark harness for
NVIDIA RTX PRO 6000 Blackwell / SM120 inference systems.

The project is focused on a narrow, upstream-relevant question:

> When new workstation Blackwell GPUs appear before the inference software stack
> has fully caught up, how do we prove whether vLLM and SGLang are selecting the
> right low-precision kernels, falling back safely, and producing reproducible
> evidence?

This is not a speed-chart repository yet. It is a reproducibility, safety, and
upstream-discipline repository: every claim must point to an artifact, a test, a
source map, or an explicit blocker.

## Problem Statement

Modern inference stacks choose among many hardware-specific backends:
FlashInfer, CUTLASS, Triton, Marlin, FP8 GEMM paths, fused MoE kernels, and
attention variants. On RTX PRO 6000 Blackwell / SM120, backend selection is
especially easy to get wrong because workstation Blackwell is not automatically
equivalent to datacenter Blackwell.

This repo tracks two concrete issues:

1. **vLLM SM120 FP4 / MXFP4 / NVFP4 backend selection**
   - Public anchor: `vllm-project/vllm#31085`
   - Focus: whether SM120 selects a native low-precision backend or falls back
     to Marlin, and whether that behavior is correct and clearly logged.

2. **SGLang RTX PRO Blackwell FP8 attention/backend/shared-memory behavior**
   - Public anchor: `sgl-project/sglang#16816`
   - Focus: whether SGLang chooses safe attention, GDN linear-attention, and
     FP8 GEMM backends on RTX PRO Blackwell without over-assuming shared-memory
     limits.

## What This Repo Contains

- GPU contention guardrails:
  - `scripts/gpu_guard.py`
  - `scripts/run_with_gpu_lock.py`
- Environment and reproducibility tools:
  - `scripts/verify_blackwell.py`
  - `scripts/collect_versions.py`
  - `scripts/check_reproducibility.py`
  - `scripts/archive_results.py`
- OpenAI-compatible serving benchmark client:
  - `benchmarks/serve_bench.py`
  - `benchmarks/summarize_results.py`
- Minimal repro wrappers:
  - `repros/vllm_mxfp4_sm120/`
  - `repros/sglang_attention_backend_sm120/`
- Upstream-ready investigation artifacts:
  - `upstream/vllm/issue_or_pr_draft.md`
  - `upstream/vllm/patch_plan.md`
  - `upstream/sglang/issue_or_pr_draft.md`
  - `upstream/sglang/patch_plan.md`

## Current Findings

These are the only findings currently supported by saved local evidence and
tests:

- The harness can collect environment metadata without initializing CUDA by
  default.
- GPU-consuming commands are designed to run through a per-GPU lock wrapper that
  records selected physical GPU IDs and before/after GPU state.
- The benchmark client can produce dry-run JSONL and summaries, and the
  summarizer marks dry-run, contended, low-sample, approximate-token, and
  unlocked rows as non-headline evidence.
- Installed vLLM `0.12.0` static selector simulation returns `MARLIN` for mocked
  SM120 MXFP4 scenarios. This is source-level evidence only, not runtime backend
  proof.
- Local SM120 CUDA device metadata was collected under the lock wrapper in a
  prior run. SGLang runtime evidence is still blocked because SGLang is not
  installed in the current environment.
- No valid real serving benchmark result exists yet. No speedup, throughput,
  correctness, or runtime backend-selection claim is made.

Generated raw results are intentionally not committed because they can contain
machine-specific paths, GPU UUIDs, process IDs, and other local metadata. The
public result index is `results/README.md`; local evidence tracking is described
in `docs/evidence_ledger.md`.

## Reproducibility

Use a repo-local virtual environment. Do not rely on the current shell's global
or private Python environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements.txt

python scripts/collect_versions.py --out results/env/versions.json
python scripts/check_reproducibility.py --out results/env/reproducibility_check.json
pytest -q
```

Detailed setup and dependency notes:

- `docs/reproducibility.md`
- `docs/dependency_matrix.md`
- `docs/model_access_plan.md`
- `docs/external_repo_setup.md`

## GPU Safety Rule

Every command that can initialize CUDA, allocate GPU memory, start a model
server, or run inference must go through:

```bash
python scripts/run_with_gpu_lock.py --gpus 0 --min-free-gb 70 --wait --label <label> -- <command>
```

Before any real GPU run:

```bash
python scripts/gpu_guard.py status
python scripts/gpu_guard.py check --gpus 0 --min-free-gb 70
```

The project does not kill, renice, suspend, or otherwise interfere with other
users' GPU processes. Contended runs are excluded from headline metrics.

## Dry-Run Validation

Dry-run path, no server and no model download:

```bash
python benchmarks/serve_bench.py \
  --dry-run \
  --framework vllm \
  --model hf-internal-testing/tiny-random-gpt2 \
  --out results/benchmarks/repro_dry_run.jsonl

python benchmarks/summarize_results.py \
  --results results/benchmarks \
  --out results/benchmarks/summary.csv \
  --markdown-out results/benchmarks/summary.md
```

Dry runs validate schema only. They are not benchmark results.

## Upstream Patch Discipline

Patch plans are intentionally conservative:

- vLLM: start with diagnostics and selector tests; do not broadly widen SM100
  checks to include SM120 without runtime evidence.
- SGLang: keep full-attention, GDN linear-attention, and FP8 GEMM changes
  separate; do not hardcode local shared-memory assumptions.

Current patch planning files:

- `upstream/vllm/patch_plan.md`
- `upstream/vllm/diff_summary.md`
- `upstream/sglang/patch_plan.md`
- `upstream/sglang/diff_summary.md`
- `docs/upstream_pr_protocol.md`

## Roadmap

1. Create clean external source checkouts for vLLM and SGLang.
2. Run one valid, uncontended, one-GPU tiny-model smoke test.
3. Capture real vLLM backend-selection logs on SM120.
4. Install/check out SGLang in an isolated environment and capture backend logs.
5. Add targeted upstream diagnostics/tests.
6. Run only minimal, evidence-driven target repros.
7. Publish upstream issue updates or PRs when runtime evidence is sufficient.

## Current Status

The repo is ready as a public scaffold and investigation harness. It is not yet
a completed benchmark report and not yet an upstream patch submission.

The most important blockers are:

- no valid real serving benchmark yet,
- no external vLLM/SGLang source checkout committed under `external/`,
- SGLang not installed in the current environment,
- no large target-model download/use approved,
- no initial upstream PR-ready patch until runtime evidence is collected.
