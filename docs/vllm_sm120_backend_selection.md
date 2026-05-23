# vLLM SM120 FP4/MXFP4/NVFP4 Backend Selection

Status: diagnostics-first investigation. No vLLM source patch is ready until a
locked SM120 hardware repro proves which native backend actually loads and runs.

## Inspected Versions

- Installed vLLM: `0.12.0` from the active Python environment's
  `site-packages/vllm` path; private absolute path redacted.
- Installed packages: `torch 2.9.0`, `triton 3.5.0`, `flashinfer-python 0.5.3`.
- `external/vllm` is a clean sparse checkout at
  `5bb8d2767a2829b56e58c68fa8f380e9e4e2bd3e`.
- Current upstream source audit: `docs/current_upstream_source_audit.md`.

Bootstrap command used for local upstream inspection:

```bash
mkdir -p external
git clone --filter=blob:none --sparse https://github.com/vllm-project/vllm.git external/vllm
git -C external/vllm sparse-checkout set \
  vllm/model_executor/layers/quantization \
  vllm/model_executor/layers/fused_moe \
  vllm/platforms \
  tests/kernels/moe \
  tests/v1/attention
git -C external/vllm checkout 5bb8d2767a2829b56e58c68fa8f380e9e4e2bd3e
git -C external/vllm rev-parse HEAD
```

## Evidence Artifacts

- Static selector probe:
  `results/repros/vllm_mxfp4_sm120/static_backend_selection_probe.json`
- vLLM dry-run command metadata:
  `results/repros/vllm_mxfp4_sm120/20260520T100045Z/command.txt`
- Latest GPU status:
  `results/gpu_status/vllm_deep_dive_status_latest.json`
- GPU eligibility checks:
  `results/gpu_status/vllm_deep_dive_gpu0_check.json`,
  `results/gpu_status/vllm_deep_dive_gpu1_check.json`

No real vLLM GPU repro was run in this pass because both GPUs had active
compute process PID `2805853`.

## Installed vLLM 0.12.0 Codepath

The installed wheel uses the legacy MXFP4 selector:

- `vllm/model_executor/layers/quantization/mxfp4.py`
  - `Mxfp4Backend`
  - `get_mxfp4_backend_with_lora()`
  - `get_mxfp4_backend(with_lora_support)`
  - `Mxfp4MoEMethod.__init__()`
- `vllm/platforms/interface.py`
  - `DeviceCapability.to_int()`
  - `Platform.has_device_capability()`
  - `Platform.is_device_capability()`
- `vllm/model_executor/layers/quantization/utils/mxfp4_utils.py`
  - `_swizzle_mxfp4()`
- `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py`
  - `is_flashinfer_fp4_cutlass_moe_available()`
  - `is_flashinfer_fp4_cutedsl_moe_available()`
  - `select_nvfp4_gemm_impl()`
- `vllm/model_executor/layers/quantization/utils/nvfp4_moe_support.py`
  - `detect_nvfp4_moe_support()`
- `vllm/model_executor/layers/quantization/modelopt.py`
  - `ModelOptNvFp4LinearMethod`
  - `ModelOptNvFp4FusedMoE`

Installed MXFP4 behavior from static probe:

- SM120 has compute capability `(12, 0)`.
- `is_device_capability(100)` is false for SM120.
- `has_device_capability(100)` is true for SM120.
- MXFP4 FlashInfer branches in `get_mxfp4_backend()` require exact SM100.
- Triton fallback requires `(9, 0) <= capability < (11, 0)`, so SM120 is excluded.
- With FlashInfer and Triton mocked available, SM120 returns `Mxfp4Backend.MARLIN`.

Installed NVFP4 helper behavior from static probe:

- FlashInfer CUTLASS FP4 helper uses `has_device_capability(100)`, so SM120 can pass
  if the FlashInfer feature probe also passes.
- FlashInfer CuTeDSL helper uses exact SM100, so SM120 does not pass that helper.
- `cutlass_fp4_supported()` delegates to a compiled vLLM op, so Python source alone
  cannot prove SM120 kernel support.

## Current Upstream HEAD Notes

Upstream HEAD has refactored GPT-OSS MXFP4 selection into an oracle path:

- `vllm/model_executor/layers/quantization/mxfp4.py`
  delegates routed-expert MXFP4 to `select_mxfp4_moe_backend()`.
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`
  defines `Mxfp4MoeBackend`, backend priority lists, explicit env handling,
  fallback behavior, and backend logging.
- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`
  defines `NvFp4MoeBackend` and NVFP4 MoE backend selection.
- `vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py`
  gates TRTLLM MXFP4 on `is_device_capability_family(100)` plus FlashInfer.
- `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py`
  explicitly allows SM90, SM100 family, and SM120 family for FlashInfer CUTLASS;
  SM110 is intentionally excluded in the source comment.
- `vllm/platforms/interface.py`
  now includes `is_device_capability_family()`, where family `100` means major
  capability `10.x`, not SM120.

This means current main may already be closer for the FlashInfer CUTLASS
MXFP4/NVFP4 path, but source inspection still cannot prove that the selected
kernel loads or produces correct output on RTX PRO 6000 Blackwell.

## Root-Cause Hypothesis

For installed vLLM `0.12.0`, the SM120 MXFP4 fallback is explained by Python
selector logic:

1. Native MXFP4 FlashInfer branches use exact SM100 checks.
2. SM120 is not exact SM100.
3. Triton MXFP4 fallback is intentionally limited to SM90 and SM100-era
   capabilities and excludes SM120.
4. The selector therefore chooses Marlin.

For upstream main, the hypothesis is narrower:

1. Some backend classes still use SM100-family checks that exclude SM120.
2. FlashInfer CUTLASS code now has an explicit SM120-family allowance.
3. The next experiment must determine whether auto-selection chooses CUTLASS,
   TRTLLM, Marlin, or raises, and whether the selected path actually runs.

## Minimal Reproducer

Dry-run only, already executed:

```bash
bash repros/vllm_mxfp4_sm120/run_repro.sh \
  --model openai/gpt-oss-20b \
  --quantization mxfp4 \
  --dry-run
```

Real one-GPU repro to run only after a guard check is eligible:

```bash
python scripts/gpu_guard.py check --gpus 1 --min-free-gb 70

python scripts/run_with_gpu_lock.py \
  --gpus 1 \
  --min-free-gb 70 \
  --wait \
  --poll-seconds 30 \
  --max-wait-seconds 600 \
  --label vllm_mxfp4_repro_gpu1 \
  -- \
  bash repros/vllm_mxfp4_sm120/run_repro.sh \
    --model <MXFP4_MODEL_ID_OR_LOCAL_PATH> \
    --tp 1 \
    --dtype auto \
    --quantization mxfp4
```

The real run must save:

- wrapper metadata under `results/gpu_runs/...`,
- `verify_blackwell.json`,
- `server_stdout.log` and `server_stderr.log`,
- `backend_grep.txt`,
- `backend_summary.json`,
- `bench_raw.jsonl`,
- benchmark summary stdout JSON,
- contention status.

## Safe Patch Plan

Do not widen exact SM100 checks to `has_device_capability(100)` in a broad patch.
That would admit SM110 and future devices into paths whose kernel support has not
been demonstrated.

Safe next patch candidates:

1. Upstream diagnostics patch:
   - log device capability,
   - log FlashInfer/CUTLASS/Triton availability probes,
   - log rejected backend candidates and reasons,
   - keep existing fallback behavior.
2. If SM120 CUTLASS runtime evidence succeeds:
   - add or preserve an explicit SM120-family branch for that specific backend,
   - add mocked selector tests for SM90, SM100, SM110, and SM120,
   - add a hardware-gated test note for SM120.
3. If SM120 falls back to Marlin:
   - improve the Marlin warning so it distinguishes "native FP4 unsupported" from
     "native path unavailable or not validated for this device/backend".

## Tests To Run

Local lightweight tests:

```bash
python scripts/vllm_mxfp4_static_probe.py \
  --out results/repros/vllm_mxfp4_sm120/static_backend_selection_probe.json \
  --upstream-head 5bb8d2767a2829b56e58c68fa8f380e9e4e2bd3e

pytest -q
```

Future upstream tests:

- mocked selector tests for MXFP4 backend selection,
- mocked NVFP4 FlashInfer helper tests,
- hardware-gated SM120 kernel smoke test,
- one short server smoke through this repo's GPU lock wrapper.
