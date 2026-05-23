# Dependency Matrix

Status: observed on this machine during the reproducibility hardening pass. This
matrix is not a lockfile; it separates the base harness from optional framework
environments.

## Current Environment Summary

- Python: `3.12.3`
- Current interpreter is in a non-repo virtual environment. Reproduction should
  use `.venv` in the repo root.
- `external/vllm`: absent
- `external/sglang`: absent
- SGLang Python packages: absent
- Version artifact: `results/env/versions.json`

## Packages

| Package | Observed version | Required / recommended | Source of truth | Location | Notes |
|---|---:|---:|---|---|---|
| Python | `3.12.3` | `>=3.10`; prefer 3.12 on this machine | `pyproject.toml` and `results/env/versions.json` | current env | Use repo-local `.venv` for reproduction. |
| requests | `2.32.4` | `>=2.32.0` | `requirements.txt` | current env | Benchmark client HTTP dependency. |
| psutil | `7.0.0` | `>=5.9.0` | `requirements.txt` | current env | Utility dependency; not GPU-specific. |
| PyYAML | `6.0.2` | `>=6.0.0` | `requirements.txt` | current env | Reserved for structured configs. |
| pandas | `2.3.1` | `>=2.0.0` | `requirements.txt` | current env | Benchmark summary CSV/Markdown support. |
| numpy | `1.26.4` | `>=1.26.0` | `requirements.txt` | current env | Statistics/schema support. |
| pytest | `9.0.2` | `>=8.0.0` | `requirements.txt` | current env | Local tests. |
| ruff | not installed | `>=0.6.0` recommended for lint only | `requirements.txt` | missing | Missing here; tests do not require it. Install in `.venv` for lint. |
| torch | `2.9.0` | optional, match CUDA 12.8 stack when probing GPU | verifier artifacts | current env | Do not use `torch.cuda` outside the GPU lock wrapper. |
| triton | `3.5.0` | optional, framework-dependent | verifier artifacts | current env | Needed for framework/runtime investigations. |
| vLLM | `0.12.0` | optional installed wheel; upstream source preferred for patches | verifier artifacts and `upstream/vllm/patch_plan.md` | current env | Installed wheel supports static probing; not a checked-out upstream source. |
| SGLang | not installed | optional, install only in isolated env | verifier artifacts and `upstream/sglang/patch_plan.md` | missing | Required before SGLang runtime repros. |
| sgl-kernel | not installed | optional, SGLang-dependent | verifier artifacts | missing | Required by some SGLang paths. |
| flashinfer-python | `0.5.3` | optional, framework-dependent | verifier artifacts | current env | Package distribution is `flashinfer-python`; import name may differ by version. |
| transformers | `4.57.6` | optional for model metadata/use | `results/env/versions.json` | current env | Do not use to download large models without approval. |
| huggingface-hub | `0.36.2` | optional for cache inspection/downloads | `results/env/versions.json` | current env | Never commit tokens/cache paths. |
| accelerate | `1.13.0` | optional | `results/env/versions.json` | current env | Not part of base harness. |
| safetensors | `0.5.3` | optional | `results/env/versions.json` | current env | Model-file support. |

## Install Order

Base harness:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements.txt
python scripts/collect_versions.py --out results/env/versions.json
python scripts/check_reproducibility.py --out results/env/reproducibility_check.json
pytest -q
```

Optional source checkouts are documented in `docs/external_repo_setup.md`.
Optional vLLM/SGLang installs should happen only inside `.venv` or a named
isolated environment, never in global Python.

## Known Risks

- The current shell relies on a non-repo virtual environment, so package
  availability is not enough for reproduction.
- vLLM is installed as a wheel but `external/vllm` is absent; upstream patch work
  still needs a source checkout.
- SGLang is absent; SGLang runtime claims remain blocked.
- `ruff` is listed in requirements but absent from the current environment.
- No lockfile exists. If exact package pinning becomes necessary, generate a
  constraints file from a reviewed `.venv`, not from hidden global state.
