# Benchmark Protocol

This protocol exists to make results maintainer-readable and interview-credible.

## Required metadata

Every benchmark run must include:

- timestamp UTC,
- git commit of this repo,
- framework name and commit/version,
- model ID/path,
- dtype / quantization,
- backend flags,
- tensor parallel size,
- GPU IDs,
- CUDA_VISIBLE_DEVICES,
- driver version,
- CUDA version,
- PyTorch version,
- Triton version,
- vLLM or SGLang version/commit,
- nvidia-smi snapshot before and after,
- GPU contention label,
- prompt/input length,
- max output tokens,
- concurrency,
- warmup count,
- trial count,
- command line.

## Required metrics

For serving benchmarks:

- time to first token (TTFT),
- total latency,
- generated token count,
- output tokens/sec per request,
- aggregate output tokens/sec,
- requests/sec,
- P50/P95/P99 latency if enough samples,
- GPU memory before/after.

For backend-selection repros:

- selected backend,
- fallback backend if any,
- relevant warning/error logs,
- shared-memory requested vs hardware limit if applicable,
- NaN/Inf output checks,
- deterministic prompt output.

## Warmup and trials

Minimum:

- 2 warmup requests for smoke tests,
- 5 measured requests for smoke tests,
- 10+ measured requests for benchmark claims,
- more if P95/P99 is claimed.

## Invalid benchmark conditions

A run is invalid for headline metrics if:

- GPU lock wrapper was not used,
- unrelated GPU processes were detected,
- environment metadata is missing,
- model/backend/dtype is ambiguous,
- only one trial was run but claims are broad,
- correctness was not checked for quantized backend changes.

## Machine-readable status

Benchmark JSON summaries must include:

- `run_status`: `real`, `dry_run`, `failed`, or `skipped`.
- `validity_status`: `valid_uncontended`, `contended`, `invalid_uncertain_*`, or `dry_run_no_server_contact`.
- `raw_out`: path to JSONL rows.
- `metadata.command`: exact benchmark client command.
- `metadata.gpu_locked`, `metadata.gpu_ids`, and `metadata.cuda_visible_devices`.
- `token_count_exact`: `true` only when completion token counts came from server usage metadata.

Dry runs are useful for schema checks but must never be used as benchmark evidence. Summaries that are missing raw JSONL, lock metadata, exact token counts, or uncontended GPU status must be marked `headline_eligible=false` by `benchmarks/summarize_results.py`.

The standalone benchmark client must not send requests to a model server unless it is launched inside `scripts/run_with_gpu_lock.py`. The `--allow-unlocked-client` escape hatch is only for non-GPU local test servers and does not produce headline-eligible evidence.

## Ramp-up matrix

Benchmarks progress in stages:

1. Stage A: one-GPU tiny/small smoke benchmark with short context, low
   concurrency, and at least 5 measured requests.
2. Stage B: one-GPU meaningful benchmark with a small model that exercises
   serving behavior, 10+ measured requests, and context lengths such as 2K and
   8K only if safe.
3. Stage C: target repro benchmark for vLLM FP4/MXFP4/NVFP4 or SGLang FP8
   backend behavior. Run only the minimal target experiment needed to validate
   the issue or patch.
4. Stage D: TP=2 benchmark only after matching TP=1 evidence is valid and both
   GPUs are eligible.

Current matrix status is documented in `docs/benchmark_matrix.md`. Dry-run rows
validate schema only and must not be cited as performance evidence.

## Reporting template

```md
# Benchmark: <framework> <model> <backend>

## Environment

- Date:
- Host:
- GPUs:
- Driver/CUDA:
- Framework commit:
- Command:

## Configuration

- dtype/quantization:
- backend flags:
- tensor parallel:
- context/input/output lengths:
- concurrency:

## Validity

- GPU lock used: yes/no
- Contention label:
- Warmups/trials:

## Results

| metric | value |
|---|---:|
| TTFT median |  |
| TPOT median |  |
| output tok/s aggregate |  |
| latency P50/P95/P99 |  |
| peak GPU memory |  |

## Interpretation

What bottleneck likely dominated? What evidence supports that? What remains uncertain?
```
