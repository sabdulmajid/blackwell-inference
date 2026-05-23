# Bootstrap Manifest

This scaffold includes:

- Codex durable instructions: `AGENTS.md`
- Optional Codex project config/subagents: `.codex/` if created locally
- Master goal and task board: `MASTER_GOAL.md`, `TASK_BOARD.md`
- GPU contention guard scripts: `scripts/gpu_guard.py`, `scripts/run_with_gpu_lock.py`
- Environment verifier: `scripts/verify_blackwell.py`
- Non-interactive Codex goal loop: `scripts/goal_loop.sh`
- Benchmark client: `benchmarks/serve_bench.py`
- vLLM/SGLang repro directories and launch scripts
- Upstream PR/issue draft templates
- Technical docs and protocols

After copying, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/gpu_guard.py status
python scripts/verify_blackwell.py --out results/env/verify_blackwell.json
```
