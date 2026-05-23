# Upstream PR / Issue Protocol

## Before opening a PR

For vLLM or SGLang, prepare:

1. Minimal reproducer.
2. Environment metadata.
3. Current upstream behavior.
4. Proposed change.
5. Correctness evidence.
6. Benchmark evidence if performance is claimed.
7. Limitations and hardware-specific caveats.
8. Tests added or explanation of hardware-gated testing.

## No Silly Mistakes Checklist

Before posting an issue update or opening an upstream PR, verify:

- [ ] No broad hardware equivalence assumption, such as treating SM120 as SM100
      or server Blackwell without backend-specific evidence.
- [ ] No hidden large-model requirement for ordinary unit tests; large models
      are manual/hardware-gated validation only.
- [ ] No stale upstream commit SHA. Re-run `git rev-parse HEAD` in the external
      checkout and include it in the draft.
- [ ] No missing environment metadata. Link the relevant `verify_blackwell.py`
      output and GPU wrapper metadata.
- [ ] No performance, speedup, latency, throughput, or regression claim without
      valid uncontended raw JSONL and summaries under `results/`.
- [ ] No patch that only works on this workstation because of hardcoded local
      paths, GPU names, memory sizes, or shared-memory constants.
- [ ] No fallback removal without tests proving the replacement path is correct.
- [ ] No uncontrolled backend forcing. Preserve explicit user overrides and
      fail clearly when no supported backend exists.
- [ ] No mixed vLLM/SGLang branch or combined attention/GEMM patch unless the
      upstream maintainer explicitly requests that scope.
- [ ] No benchmark artifacts or model files committed into upstream source
      branches unless the upstream project specifically expects them.
- [ ] No claim that a dry-run, static probe, or contended run proves runtime
      backend selection.
- [ ] No dirty external checkout ignored. Inspect `git status --short` and
      `git diff --stat` before editing.

Reject the patch if any checkbox cannot be satisfied or explicitly explained in
the PR body.

## PR title examples

vLLM:

- `Add guarded SM120 MXFP4 backend selection handling for RTX PRO Blackwell`
- `Improve NVFP4/MXFP4 backend diagnostics for SM120 RTX Blackwell`

SGLang:

- `Allow validated FlashInfer backend path for RTX PRO Blackwell hybrid GDN models`
- `Improve Blackwell attention backend selection when Triton shared memory exceeds SM120 limit`

## PR body template

```md
## Summary

<One paragraph.>

## Problem

- Hardware:
- Model:
- Framework commit:
- Current behavior:
- Expected behavior:

## Root cause / hypothesis

<Exact codepath and why current behavior occurs.>

## Change

<Minimal patch. Explain why it is safe.>

## Validation

### Environment

<Attach `verify_blackwell.json` summary.>

### Reproducer

```bash
<command>
```

### Before

<logs>

### After

<logs>

### Correctness

<checks>

### Performance

<only if benchmarked>

## Risks / limitations

- <e.g. SM120 not a superset of SM100; fallback retained.>

## Tests

- [ ] unit tests
- [ ] hardware-gated manual test
- [ ] benchmark smoke test
```

## Issue-comment template

Use if a PR is not appropriate yet:

```md
I tested this on RTX PRO 6000 Blackwell / SM120 with the following environment:

<env summary>

Current behavior on <commit>:

<logs>

Minimal repro:

<command>

Findings:

1. ...
2. ...
3. ...

Possible patch direction:

<careful, non-overclaiming suggestion>

Artifacts:

- repro log:
- environment:
- benchmark summary:
```
