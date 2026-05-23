# vLLM SM120 MXFP4/NVFP4 Reproducer

Goal: reproduce and diagnose whether current vLLM selects an appropriate backend for SM120 RTX PRO 6000 Blackwell when running MXFP4/NVFP4/FP4 models.

## Current public anchor

`vllm-project/vllm#31085` reports that SM120 / RTX 6000 Pro Blackwell compute capability `(12, 0)` is not recognized in MXFP4 backend selection, causing fallback to Marlin rather than native NVFP4 kernels.

## What this repro must collect

- exact vLLM commit/version,
- exact model path/ID,
- GPU compute capability,
- chosen backend,
- fallback backend if any,
- warning/error logs,
- correctness sanity output,
- GPU status before/after,
- benchmark result if a server launches.

## Run pattern

```bash
python scripts/run_with_gpu_lock.py --gpus 0 --min-free-gb 70 --wait --label vllm_mxfp4_repro -- \
  bash repros/vllm_mxfp4_sm120/run_repro.sh --model <MODEL_ID_OR_PATH> --tp 1
```

For 2-GPU tensor parallel:

```bash
python scripts/run_with_gpu_lock.py --gpus 0,1 --min-free-gb 70 --wait --label vllm_mxfp4_repro_tp2 -- \
  bash repros/vllm_mxfp4_sm120/run_repro.sh --model <MODEL_ID_OR_PATH> --tp 2
```

## Expected result categories

- `upstream_fixed`: native/expected backend selected on current main; document evidence.
- `fallback_expected`: fallback occurs but is correct due to unsupported native SM120 path; document why.
- `bug_reproduced`: fallback/error occurs unexpectedly; prepare patch or issue update.
- `blocked`: model access/install/hardware constraint; document exact blocker.
