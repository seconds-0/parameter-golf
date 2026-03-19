# Repository Guidelines

## Project Structure & Module Organization
Core training entry points live at the repository root: `train_gpt.py` for CUDA and `train_gpt_mlx.py` for Apple Silicon MLX. Experiment orchestration lives under `experiments/`, with YAML configs in `experiments/configs/`, machine definitions in `experiments/machines.yaml`, automation scripts in `experiments/scripts/`, and infra tests in `experiments/scripts/tests/`. Dataset and tokenizer helpers live in `data/`. Reproducible submissions belong in `records/track_10min_16mb/` or `records/track_non_record_16mb/`, with one self-contained folder per run.

## Experiment Strategy & Tracking
`AGENTS.md` is for stable repository orientation only; it should not be treated as the source of current experiment state.

- **Read `docs/tracker.md` first** to see current progress, next tasks, and what is blocked.
- Read `docs/experiment_plan_prd.md` for the full strategy and promotion logic.
- Read `docs/tracks/track_{a-f}_*.md` for per-track thesis, experiments, decision rules, and learnings.

Key principles that govern all experiment work:
- **Post-roundtrip is the only real metric.** Never promote on pre-quant improvement alone. Track `Δpq` (post-roundtrip delta vs baseline, negative=better) and `qgap` (post-quant minus pre-quant bpb, smaller=better).
- **Sequential halving, not grids.** One seed at cheapest proxy, keep top third, escalate survivors only.
- **Four proxy levels:** P0 (smoke), P1 (1xH100 ~5min), P2 (1xH100 ~15-20min), P3 (8xH100 ~2min timing only).
- **Experiment configs** follow naming: `experiments/configs/{phase}_{id}_{name}.yaml` with proxy presets at `proxy_p{0-3}_*.yaml`.
- **Budget discipline:** Conservative plan ~12-15 H100-hours. Phase caps enforced. No 8xH100 until a candidate has 2 P2 seeds with mean Δpq ≤ -0.006.

## Build, Test, and Development Commands
Create an environment and install dependencies with `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

Use these common commands:
- `make test`: runs the infra regression suite in `experiments/scripts/tests/test_infra.py`.
- `make validate CONFIG=experiments/configs/baseline.yaml`: validates a run config before launch.
- `make preflight HOST=<host>`: checks remote machine readiness.
- `make run CONFIG=experiments/configs/baseline.yaml HOST=<host>`: launches a configured experiment.
- `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`: downloads a small local dataset subset.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, type hints where helpful, `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for module-level constants, and short explanatory comments only when the code is not obvious. Keep scripts direct and readable; `train_gpt.py` explicitly warns against growing core training scripts without clear need. Prefer standard library tools plus the dependencies already listed in `requirements.txt`.

## Testing Guidelines
Pytest is the test framework. Add or update tests in `experiments/scripts/tests/test_infra.py` when changing launch, config, parsing, cost, or watchdog behavior. Name tests `test_<behavior>()` and keep fixtures minimal. Run `make test` before opening a PR; if you add a new config flow, also run `make validate CONFIG=...` against the affected YAML.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Fix review issues: ...` or `Add optional wandb integration to train_gpt.py`. Keep commits focused and descriptive. PRs should explain the motivation, list the commands you ran, and link any relevant issue or discussion. For benchmark or challenge submissions, add a new folder under `records/` with `README.md`, `submission.json`, `train.log`, and any required training code; do not replace prior records.
