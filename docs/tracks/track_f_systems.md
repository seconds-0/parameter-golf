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
- No model conclusion is trustworthy until E02 passes (within 0.003 bpb of baseline); E00 and E01 exist only to validate the harness
- Every run must emit tok_s, train_tokens_seen, prequant exact, and qgap
- Config-driven paths — no hardcoded SP-1024 assumptions
- Budget discipline: phase caps, conservative plan (~12-15 H100-hours)
- All new experiment knobs as env vars (auto-discovered by config_utils)

## Status
Trusted for launch/measurement on the training path, but not yet trusted for standalone saved-artifact replay. `P-01`, `P-02`, `P-03`, `P-04`, `P-05`, `P-06`, `X-01`, `X-02`, `E00`, `E01`, and `E02` are done. The next work in sequence is now a focused fresh-process replay bughunt, because two fresh instrumented runs showed trainer-side reload is healthy while standalone replay is still wrong.

## Learnings
- MLX lazy eval bug fixed (mx.synchronize → mx.eval)
- 53 infra tests passing
- Codex + Gemini review hardened the replay utility, but a deeper fresh-process reconstruction bug remains
- `E00` succeeded on `proxy_p0_smoke-20260319-082117-62aaaea6` with post-roundtrip `val_bpb=2.59277612`, `qgap_bpb=0.00857417`, and `artifact_slack_bytes=9,350,400`
- The first live `1xH100` run caught a launcher bug: `launch.py run` was defaulting to `8` GPUs instead of the machine's configured GPU count
- Triton requires a tiny compile toolchain on remote hosts; preflight now verifies `Python.h` and `gcc` before launch
- Watchdog stall detection must respect log cadence (`TRAIN_LOG_EVERY`), not raw per-step time, or it will kill healthy runs during final eval
- `compare.py` and `launch.py status` now surface postquant/prequant metrics, `qgap`, artifact slack, step time, and throughput directly
- `E01` passed after stabilizing the Vast host on `Python 3.11.15`; the successful seed pair landed at postquant `val_bpb` 1.38651817 and 1.38373201, for a spread of 0.00278616
- `E01` establishes the first trusted P1 control band: mean postquant `val_bpb=1.38512509`, mean `qgap_bpb=0.00249501`, mean step time `≈336.8 ms`, and throughput `≈1.56M tok/s`
- Fresh `8xH100` bring-up exposed a remote-interpreter mismatch: the harness was checking bare `python3` while the host setup used a shared repo `.venv`; preflight, dependency checks, auto-downloads, and launches now all prefer that shared `.venv/bin/python`
- Fresh `8xH100` bring-up also exposed a data footgun: `cached_challenge_fineweb.py` was defaulting to `80` train shards, which is fine for proxies but unsafe for baseline reproduction; it now defaults to the full published train split and uses `--train-shards` only for explicit proxy prefixes
- `E02` passed on `baseline_repro-20260319-093041-82dade88` with exact post-roundtrip `val_bpb=1.22628807`, prequant `val_bpb=1.21919389`, `qgap_bpb=0.00709417`, `total_submission_bytes=15,861,240`, and `artifact_slack_bytes=138,760`
- The `E02` runtime profile matches the published baseline closely enough to trust future deltas: `step_avg_ms=43.55`, `tok_s=12.04M`, and the exact post-roundtrip gap vs the published `1.22436570` is only `+0.00192237`
- `export_eval.py` now prefers the checkpoint's neighboring manifest env and trainer snapshot, scrubs ambient trainer env vars before import, and writes `final_model.export_eval.int8.ptz` by default so replay debugging does not overwrite a trusted artifact
- Fresh 1xH100 replay checks proved the remaining bug is deeper than exporter config drift: the saved `E02` raw checkpoint replayed at `prequant_val_bpb=1.81346210` / `postquant_val_bpb=1.82621440`, and the saved `E01` raw checkpoint replayed at `1.79180195` / `1.79691088`
- The trainer now logs `uncompiled_check`, `checkpoint_save_verify`, `reloaded_prequant_exact`, and `reloaded_int8_zlib_roundtrip_exact`, which gives us an in-process proof path for compiled-vs-eager drift and raw/quantized save fidelity
- `parse_log.py`, `compare.py`, and `launch.py status` now expose the replay-trust deltas directly, so a single instrumented GPU run can decide the next fix with much less ambiguity
- The fresh proof run `proxy_p1_fast-20260319-185044-b42cf8f0` did exactly that separation: inside the trainer, the raw checkpoint and quantized artifact stayed within about `+0.0006` bpb of the live end-of-run metrics, and the raw checkpoint save verified byte-for-byte against the live state dict
- Replaying that same fresh checkpoint in a separate process still failed badly via `export_eval.py` (`1.78986763/1.79694755`), so the systems blocker is no longer “trainer serialization is suspicious”; it is now “fresh-process model reconstruction or snapshot import does not match the live trainer path”
- `Rotary.inv_freq` is now serialized in checkpoints, and a second fresh proof run `proxy_p1_fast-20260319-192237-39db664f` still reproduced the same standalone replay failure on the original remote artifact, so the blocker is narrower: it is not just a missing non-persistent RoPE buffer
