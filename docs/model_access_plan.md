# Model Access Plan

This plan avoids accidental large downloads. Commands below show shape only;
do not run them unless storage, access, and GPU eligibility are confirmed.

## Cache And Credential Rules

- Do not commit Hugging Face tokens or cache paths.
- Prefer explicit cache roots such as `HF_HOME=/path/to/reviewed/cache` in local
  shell history or scheduler config, not in committed files.
- Before using a model, record whether it is local, cached, gated, or requires a
  download.
- If a model is not cached, estimate size from model card or local metadata
  before running a serving command.
- Large target models require human approval before download/use.

Safe local cache inspection shape:

```bash
huggingface-cli scan-cache
```

Do not paste token values into logs, docs, or issue drafts.

## Candidate Matrix

| Category | Candidate | Expected size risk | Gated/access risk | Purpose | Command shape |
|---|---|---:|---:|---|---|
| Tiny smoke | `hf-internal-testing/tiny-random-gpt2` | usually tiny; verify first | usually public | Benchmark client/server smoke only | `python benchmarks/serve_bench.py --dry-run --framework vllm --model hf-internal-testing/tiny-random-gpt2 --out results/benchmarks/tiny_dry.jsonl` |
| Tiny smoke | `sshleifer/tiny-gpt2` | usually tiny; verify first | usually public | Alternate tiny causal LM smoke | same as above with model changed |
| Small meaningful | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | small but nontrivial, likely multiple GB | usually public | One-GPU serving behavior after tiny smoke | run only after cache/size approval |
| vLLM target | `<MXFP4_MODEL_ID_OR_LOCAL_PATH>` | unknown to very large | may be gated | Exercise FP4/MXFP4/NVFP4 backend selection | `bash repros/vllm_mxfp4_sm120/run_repro.sh --model <MODEL> --quantization mxfp4 --dry-run` |
| vLLM target | `openai/gpt-oss-20b` | large; verify model-card size | access/license unknown here | Candidate from investigation notes; not a smoke model | approval required before download/use |
| SGLang target | `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` | very large | may require terms/storage review | Target FP8 hybrid GDN repro | approval required before download/use |

## Next Safe Model Steps

1. Run dry-run command construction only:

   ```bash
   bash repros/vllm_mxfp4_sm120/run_repro.sh \
     --model hf-internal-testing/tiny-random-gpt2 \
     --dry-run
   ```

2. Inspect local cache manually before any real model use:

   ```bash
   huggingface-cli scan-cache
   ```

3. If the tiny model is not cached, ask for approval before downloading even if
   it is expected to be small.

4. If model access fails:

   - save the exact error under `docs/failed_attempts.md`,
   - keep the run out of benchmark summaries,
   - fall back to dry-run/schema validation or static backend-selection tests.

## Target Repro Approval Gate

Before running target vLLM/SGLang models, document:

- model ID or local path,
- expected disk footprint,
- whether access is gated,
- cache location, without tokens,
- exact locked command,
- selected physical GPU ID,
- expected run duration,
- output artifact paths.
