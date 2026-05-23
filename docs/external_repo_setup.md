# External Repo Setup

Status: external upstream source trees now exist as clean sparse checkouts for
source inspection. They remain ignored by the main repository and should not be
committed into `blackwell-inference`.

Do not download models or run GPU workloads as part of these steps.

## vLLM

Current inspected source:

```text
5bb8d2767a2829b56e58c68fa8f380e9e4e2bd3e external/vllm
```

Exact source checkout command:

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
git -C external/vllm rev-parse HEAD | tee results/external_commits/vllm_commit.txt
```

Create a branch only when writing a patch:

```bash
git -C external/vllm switch -c blackwell-sm120-fp4-diagnostics
```

Before editing, run:

```bash
git -C external/vllm status --short
git -C external/vllm diff --stat
```

## SGLang

Current inspected source:

```text
a5a64a311a39b153d1e4d3d6bcb67e77cdc9aeae external/sglang
```

Exact source checkout command:

```bash
mkdir -p external
git clone --filter=blob:none --sparse https://github.com/sgl-project/sglang.git external/sglang
git -C external/sglang sparse-checkout set --no-cone \
  python/sglang/srt/server_args.py \
  python/sglang/srt/model_executor/** \
  python/sglang/srt/layers/attention/** \
  python/sglang/srt/layers/quantization/** \
  python/sglang/srt/utils/** \
  python/sglang/srt/configs/** \
  python/sglang/srt/models/** \
  test/**
git -C external/sglang checkout a5a64a311a39b153d1e4d3d6bcb67e77cdc9aeae
git -C external/sglang rev-parse HEAD | tee results/external_commits/sglang_commit.txt
```

Create a branch only when writing a patch:

```bash
git -C external/sglang switch -c blackwell-sm120-fp8-attn-diagnostics
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
