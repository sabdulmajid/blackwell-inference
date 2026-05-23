# SGLang Diff Summary

Status: no external SGLang checkout exists and no upstream source diff was
created in this pass.

## External Repo

- Path: `external/sglang`
- Exists: no
- Local branch: none
- Local commit: none
- Local dirty status: not applicable
- Diff stat: none

## Intended Branch

```bash
git -C external/sglang switch -c blackwell-sm120-fp8-attn-diagnostics
```

## Intended Diff Shape

The first acceptable diff should be diagnostics and dispatch-test oriented:

- log full-attention, linear-attention decode/prefill, and FP8 GEMM choices,
- log queried device capability and shared-memory limits near failure points,
- add CPU/static tests for backend flag normalization and hybrid GDN dispatch,
- avoid broad SM100/SM120 equivalence assumptions,
- avoid result files or model artifacts.

Any future `git diff --stat` should be pasted here before opening a PR.
