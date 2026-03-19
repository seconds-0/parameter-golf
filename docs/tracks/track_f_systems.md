# Track F: Systems & Harness Discipline

## Thesis
The experiment infrastructure must be trustworthy before any experimental results can be trusted. Telemetry, cheap proxies, promotion logic, and code-size hygiene are the foundation. The mistake would be treating the existing orchestration as permission to launch undisciplined sweeps.

## Work Items
- **P-01 through P-06**: Prerequisites (see tracker.md)
- **X-01 through X-07**: Proxy infrastructure
- **E00**: Baseline P0 smoke
- **E01**: Baseline P1 control
- **E02**: Baseline 8xH100 reproduction

## Key Principles
- No experiment is trustworthy until E02 passes (within 0.003 bpb of baseline)
- Every run must emit tok_s, train_tokens_seen, prequant exact, and qgap
- Config-driven paths — no hardcoded SP-1024 assumptions
- Budget discipline: phase caps, conservative plan (~12-15 H100-hours)
- All new experiment knobs as env vars (auto-discovered by config_utils)

## Status
In progress. Infrastructure built (launch.py, watchdog, preflight, etc.). Prerequisites P-01..P-06 not yet started.

## Learnings
- MLX lazy eval bug fixed (mx.synchronize → mx.eval)
- 33 infra tests passing
- Codex + Gemini reviews completed, issues fixed
