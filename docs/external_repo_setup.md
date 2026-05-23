# External Repo Setup

Status: external upstream source trees are absent in this checkout. These
commands are recorded for the next patch phase; they were not run during the
patch-discipline pass.

Do not download models or run GPU workloads as part of these steps.

## vLLM

Current remote `main` observed with `git ls-remote`:

```text
87e31455b056c6ce59bf5dcb3c622155431851db refs/heads/main
```

Exact source checkout command:

```bash
mkdir -p external
git clone --filter=blob:none --sparse https://github.com/vllm-project/vllm.git external/vllm
git -C external/vllm sparse-checkout set \
  vllm/model_executor/layers/quantization \
  vllm/model_executor/layers/fused_moe \
  vllm/platforms \
  tests/kernels/moe
git -C external/vllm checkout 87e31455b056c6ce59bf5dcb3c622155431851db
git -C external/vllm switch -c blackwell-sm120-fp4-diagnostics
git -C external/vllm rev-parse HEAD | tee results/external_commits/vllm_commit.txt
```

Before editing, run:

```bash
git -C external/vllm status --short
git -C external/vllm diff --stat
```

## SGLang

Current remote `main` observed with `git ls-remote`:

```text
1bd4f94598a621cf5e8c27686311e92134e9edb0 refs/heads/main
```

Exact source checkout command:

```bash
mkdir -p external
git clone --filter=blob:none --sparse https://github.com/sgl-project/sglang.git external/sglang
git -C external/sglang sparse-checkout set \
  python/sglang/srt/server_args.py \
  python/sglang/srt/model_executor \
  python/sglang/srt/layers/attention \
  python/sglang/srt/layers/quantization \
  python/sglang/srt/utils \
  python/sglang/srt/configs \
  python/sglang/srt/models \
  test
git -C external/sglang checkout 1bd4f94598a621cf5e8c27686311e92134e9edb0
git -C external/sglang switch -c blackwell-sm120-fp8-attn-diagnostics
git -C external/sglang rev-parse HEAD | tee results/external_commits/sglang_commit.txt
```

Before editing, run:

```bash
git -C external/sglang status --short
git -C external/sglang diff --stat
```

## Branch Discipline

- Keep vLLM and SGLang work on separate branches.
- Do not commit `results/` artifacts inside upstream repos.
- Prefer diagnostics and targeted selector tests before capability widening.
- Rebase or recreate the branch rather than stacking unrelated experiments.
- If an upstream checkout is dirty, inspect and document the diff before making
  any changes.
