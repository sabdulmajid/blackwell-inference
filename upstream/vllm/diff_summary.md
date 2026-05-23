# vLLM Diff Summary

Status: a clean external vLLM checkout exists, but no upstream source diff was
created in this pass.

## External Repo

- Path: `external/vllm`
- Exists: yes
- Local branch: detached HEAD
- Local commit: `5bb8d2767a2829b56e58c68fa8f380e9e4e2bd3e`
- Local dirty status: clean
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
