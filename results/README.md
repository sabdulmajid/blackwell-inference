# Results Artifact Index

This directory contains generated evidence. Current benchmark evidence is
schema/dry-run only unless explicitly marked real and `valid_uncontended`.

Full post-queue audit ledger: `docs/evidence_ledger.md`.

Current upstream source audit: `docs/current_upstream_source_audit.md`.

## Environment

- `results/env/verify_blackwell.json`
- `results/env/benchmark_matrix_verify_blackwell.json`
- `results/env/sglang_deep_dive_verify_blackwell.json`
- `results/env/versions.json`
- `results/env/reproducibility_check.json`

## GPU Status And Wrapper Runs

- `results/gpu_status/benchmark_matrix_pre_status.json`
- `results/gpu_status/benchmark_matrix_gpu0_check.json`
- `results/gpu_status/benchmark_matrix_gpu1_check.json`
- `results/gpu_status/hardware_sentinel_20260520_current_status.json`
- `results/gpu_status/hardware_sentinel_20260520_gpu0_check.json`
- `results/gpu_status/hardware_sentinel_20260520_gpu1_check.json`
- `results/gpu_status/benchmark_matrix_final_status.json`
- `results/gpu_status/post_queue_audit_status.json`
- `results/gpu_status/impact_resume_status.json`
- `results/gpu_runs/20260520T100750Z_sglang_device_probe/`

Latest impact-pass status: both GPUs had active Python work. No real serving
benchmark or runtime repro was launched in that state.

## Benchmarks

- `results/benchmarks/summary.csv`
- `results/benchmarks/summary.md`
- `results/benchmarks/20260520T103000Z_stage_a_vllm_tiny_dry_run.jsonl`
- `results/benchmarks/20260520T103000Z_stage_a_vllm_tiny_dry_run.summary.json`
- `results/benchmarks/20260520T103001Z_stage_a_sglang_tiny_dry_run.jsonl`
- `results/benchmarks/20260520T103001Z_stage_a_sglang_tiny_dry_run.summary.json`

All current benchmark rows are dry runs and are not headline-eligible.

## Tests

- `results/tests/post_queue_audit_pytest.txt`
- `results/tests/upstream_patch_discipline_pytest.txt`
- `results/tests/reproducibility_pytest.txt`
- `results/tests/impact_source_audit_pytest.txt`

Latest lightweight test result: `29 passed, 1 skipped in 0.77s` from
`results/tests/impact_source_audit_pytest.txt`.

Latest reproducibility check: `results/env/reproducibility_check.json`, status
`ok` with 7 documented warnings for optional/framework gaps and non-repo current
virtualenv state.

## vLLM Repro Evidence

- `results/repros/vllm_mxfp4_sm120/static_backend_selection_probe.json`
- `results/repros/vllm_mxfp4_sm120/20260520T100045Z/command.txt`

No real vLLM server benchmark or target MXFP4/NVFP4 runtime repro exists yet.

## SGLang Repro Evidence

- `results/repros/sglang_attention_backend_sm120/sglang_triton_dry_run_20260520T101900Z/command.txt`
- `results/repros/sglang_attention_backend_sm120/sglang_flashinfer_decode_dry_run_20260520T101901Z/command.txt`
- `results/gpu_runs/20260520T100750Z_sglang_device_probe/verify_blackwell_probe_cuda.json`

No real SGLang runtime repro exists yet because SGLang is not installed locally.

## External Source Inspection

- vLLM current upstream sparse checkout inspected at
  `5bb8d2767a2829b56e58c68fa8f380e9e4e2bd3e`.
- SGLang current upstream sparse checkout inspected at
  `a5a64a311a39b153d1e4d3d6bcb67e77cdc9aeae`.

These checkouts live under ignored `external/` directories. They are source
inputs, not committed result artifacts.
