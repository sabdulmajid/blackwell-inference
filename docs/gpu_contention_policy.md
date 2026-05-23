# GPU Contention Policy

The 2x NVIDIA RTX PRO 6000 Blackwell GPUs are shared resources. Other users/processes may use them. Benchmarks are invalid if they silently run during contention.

## Rules

1. Never assume GPUs are free.
2. Never kill another user's process.
3. Every GPU-consuming command must run through `scripts/run_with_gpu_lock.py`.
4. Every benchmark must record GPU status before and after.
5. Exclude contended runs from headline metrics unless the experiment is explicitly about contention.
6. Serialize GPU-heavy subagents. Do not let multiple Codex subagents launch independent GPU jobs.
7. Prefer one-GPU smoke tests before two-GPU benchmarks.
8. Use stable benchmark windows when other users are unlikely to use the GPUs.

`scripts/verify_blackwell.py` is metadata-only by default. Its `--probe-cuda` mode imports PyTorch and queries `torch.cuda`; run that mode only through `scripts/run_with_gpu_lock.py`.

## Locking model

`scripts/run_with_gpu_lock.py` uses a host-local file lock for declared GPU IDs and checks `nvidia-smi` before running the command.

This protects against this project launching overlapping jobs. It cannot prevent unrelated users from starting jobs. Therefore the wrapper also checks live `nvidia-smi` state before and after the command.

For bounded smoke attempts, use `--max-wait-seconds` so an occupied GPU produces a structured `wait_timeout` record instead of an unbounded wait.

If a guard check shows active external processes on the target GPU or free memory
below the run threshold, do not start a real serving benchmark. Save the status
artifact, mark the run blocked or contended, and retry later.

## Benchmark validity labels

Each run should be labeled:

- `valid_uncontended`: no unrelated GPU processes and sufficient free VRAM.
- `valid_self_contended`: intentional multi-process experiment from this project only.
- `invalid_contended`: unrelated GPU users/processes detected.
- `invalid_uncertain`: missing GPU status, missing environment metadata, or wrapper not used.

Only `valid_uncontended` and explicitly explained `valid_self_contended` runs may support headline performance claims.

## Recommended thresholds for 96GB GPUs

For smoke tests:

- `--min-free-gb 20`

For one-GPU mid/large model tests:

- `--min-free-gb 70`

For two-GPU large model tests:

- `--min-free-gb 70` per GPU

For long-context tests:

- `--min-free-gb 85` per GPU if practical

Adjust thresholds only with an explanation in the result metadata.

## Examples

```bash
python scripts/run_with_gpu_lock.py --gpus 0 --min-free-gb 70 -- \
  python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct
```

```bash
python scripts/run_with_gpu_lock.py --gpus 0,1 --min-free-gb 70 -- \
  bash repros/vllm_mxfp4_sm120/run_repro.sh --model /models/gpt-oss-120b
```

## Codex instructions

When planning GPU work, Codex must:

1. Ask `hardware_sentinel` or run a read-only status check.
2. Use a single command wrapped by `run_with_gpu_lock.py`.
3. Save logs in `results/`.
4. Update `TASK_BOARD.md` with exact evidence paths.
5. Never start a second GPU job while one is running.
