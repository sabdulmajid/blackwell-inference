# vLLM Diff Summary

Status: no external vLLM checkout exists and no upstream source diff was created
in this pass.

## External Repo

- Path: `external/vllm`
- Exists: no
- Local branch: none
- Local commit: none
- Local dirty status: not applicable
- Diff stat: none

## Intended Branch

```bash
git -C external/vllm switch -c blackwell-sm120-fp4-diagnostics
```

## Intended Diff Shape

The first acceptable diff should be small and diagnostics-oriented:

- selector logging in the MXFP4/NVFP4 oracle path,
- mocked selector tests,
- no broad hardware capability widening,
- no benchmark result files,
- no model artifacts.

Any future `git diff --stat` should be pasted here before opening a PR.
