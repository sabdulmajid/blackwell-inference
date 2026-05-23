.PHONY: verify verify-cuda gpu-status test lint summarize

verify:
	python scripts/verify_blackwell.py --out results/env/verify_blackwell.json

verify-cuda:
	python scripts/run_with_gpu_lock.py --gpus 0 --min-free-gb 70 --wait --label verify_blackwell_cuda -- python scripts/verify_blackwell.py --probe-cuda --out results/env/verify_blackwell_cuda.json

gpu-status:
	python scripts/gpu_guard.py status

test:
	pytest -q

lint:
	ruff check .

summarize:
	python benchmarks/summarize_results.py --results results --out results/benchmarks/summary.csv --markdown-out results/benchmarks/summary.md
