# vLLM Patch Plan: SM120 FP4 / MXFP4 / NVFP4 Backend Selection

Status: not PR-ready. No external vLLM checkout exists in this workspace, so no
upstream branch or source patch was created in this pass.

## Upstream Commit

- Patch target: vLLM `main`
  `87e31455b056c6ce59bf5dcb3c622155431851db`
- Source checkout: missing (`external/vllm` absent)
- Installed package evidence: vLLM `0.12.0`

The installed-package evidence and current upstream source map are materially
different. Revalidate the codepath after checking out the target commit.

## External Repo Status

| Field | Status |
|---|---|
| Path | `external/vllm` |
| Exists | no |
| Remote URL | blocked until checkout; intended `https://github.com/vllm-project/vllm.git` |
| Current branch | none |
| Commit SHA | none locally; remote target `87e31455b056c6ce59bf5dcb3c622155431851db` |
| Dirty status | not applicable |
| Untracked files | not applicable |
| Existing diff | none |
| Relevant tests | blocked until checkout |

Setup command: `docs/external_repo_setup.md`.

## Files And Functions Implicated

Installed vLLM `0.12.0`:

- `vllm/model_executor/layers/quantization/mxfp4.py`
  - `get_mxfp4_backend_with_lora()`
  - `get_mxfp4_backend(with_lora_support)`
  - `Mxfp4MoEMethod.__init__()`
- `vllm/platforms/interface.py`
  - `Platform.has_device_capability()`
  - `Platform.is_device_capability()`
- `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py`
  - `is_flashinfer_fp4_cutlass_moe_available()`
  - `is_flashinfer_fp4_cutedsl_moe_available()`
  - `select_nvfp4_gemm_impl()`
- `vllm/model_executor/layers/quantization/utils/nvfp4_moe_support.py`
  - `detect_nvfp4_moe_support()`
- `vllm/model_executor/layers/quantization/modelopt.py`
  - `ModelOptNvFp4LinearMethod`
  - `ModelOptNvFp4FusedMoE`

Current upstream target:

- `vllm/model_executor/layers/quantization/mxfp4.py`
  - delegation to `select_mxfp4_moe_backend()`
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`
  - `Mxfp4MoeBackend`
  - backend priority order and fallback logging
- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`
  - `NvFp4MoeBackend`
  - backend selection and env override handling
- `vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py`
  - SM100-family TRTLLM MXFP4 gating
- `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py`
  - FlashInfer CUTLASS capability allowance, including explicit SM120-family
    support per prior source map
- `vllm/platforms/interface.py`
  - capability helpers, especially exact capability vs capability family

## Minimal Patch Objective

Do not widen kernel eligibility yet. The first acceptable upstream patch is a
diagnostics and selector-test patch that makes backend decisions reviewable:

1. Emit structured or clearly parseable log lines for MXFP4/NVFP4 backend
   candidates, rejection reasons, device capability, and env overrides.
2. Add mocked selector tests for SM90, SM100, SM110, and SM120 behavior.
3. Preserve all existing fallback behavior.

Only after a locked runtime repro proves a specific SM120 backend works should a
second patch add a narrow SM120 eligibility branch for that backend.

## Alternatives Considered

- Broadly replace exact SM100 checks with `has_device_capability(100)`.
  Rejected because it may admit SM110 or future devices into unverified paths.
- Force Marlin on SM120.
  Rejected because fallback may be safe but this would hide native backend
  availability and would need correctness/performance evidence.
- Force FlashInfer CUTLASS on SM120.
  Rejected until a hardware repro proves kernel load and output correctness.

## Why This Is Not Overbroad

The planned first patch changes observability and unit-level selector coverage,
not hardware eligibility. It does not claim SM120 is equivalent to SM100 or B200,
and it keeps unsupported paths on their current fallbacks.

## Evidence Available

- Static installed-package probe:
  `results/repros/vllm_mxfp4_sm120/static_backend_selection_probe.json`
- vLLM repro dry-run command:
  `results/repros/vllm_mxfp4_sm120/20260520T100045Z/command.txt`
- Evidence ledger:
  `docs/evidence_ledger.md`
- Technical map:
  `docs/vllm_sm120_backend_selection.md`

Confirmed facts:

- Installed vLLM `0.12.0` selector simulation returns `MARLIN` for mocked SM120
  MXFP4 scenarios.
- No real vLLM runtime repro or benchmark exists yet.

## Evidence Missing

- Checked-out upstream source commit under `external/vllm`.
- Locked one-GPU runtime logs on SM120.
- Selected backend evidence from a real FP4/MXFP4/NVFP4 model load.
- Correctness output for deterministic prompts.
- Any valid benchmark or performance evidence.

## Test Plan

Local harness tests:

```bash
pytest -q
```

After external checkout:

```bash
git -C external/vllm status --short
git -C external/vllm diff --stat
python -m pytest tests/kernels/moe -q -k "mxfp4 or nvfp4 or backend"
```

Only run external tests that are CPU/static or clearly dependency-ready. Any
hardware/model test must be launched through this repo's GPU lock wrapper, not
from the external repo directly.

## PR Risk Assessment

Risk is low for diagnostics-only changes if log noise is controlled and tests do
not require GPUs or large models. Risk becomes high if the patch changes
capability gates before runtime evidence exists.

Reject a vLLM patch if it:

- replaces exact SM100 checks with broad `has_device_capability(100)` without
  backend-specific SM120 evidence,
- claims runtime backend selection from the static probe,
- removes Marlin or another fallback without tests,
- requires a large model for ordinary CI,
- combines diagnostics, capability widening, and performance claims in one diff,
  or
- includes benchmark artifacts in the upstream branch.

## Rollback / Fallback Behavior

The first patch should be fully revertible without changing selected backends.
Fallback behavior must remain exactly as upstream currently defines it.

## Submission Readiness

Not ready for PR. Ready to create an upstream branch only after
`external/vllm` is checked out at the target commit. A diagnostics-only PR may
be possible before a large-model repro, but any hardware-eligibility patch needs
one more locked SM120 experiment.
