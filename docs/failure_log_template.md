# Failure Log Template

Create one failure log per meaningful failure under `results/failures/`.

```md
# Failure: <short name>

## Timestamp

## Command

```bash
...
```

## Environment

- Repo commit:
- Framework commit:
- CUDA_VISIBLE_DEVICES:
- GPUs:
- Driver/CUDA:
- PyTorch/Triton:

## GPU status before

```text
...
```

## Expected behavior

## Actual behavior

## Stack trace / logs

```text
...
```

## Classification

- install issue / model access / backend selection / kernel failure / shared memory / OOM / correctness / performance / contention / unknown

## Hypotheses

1. ...
2. ...
3. ...

## Next experiment

## Status

open / resolved / blocked
```
