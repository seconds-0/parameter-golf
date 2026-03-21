# Track F: Systems & Harness Discipline

## Thesis
The experiment infrastructure must be trustworthy before any experimental results can be trusted. Telemetry, cheap proxies, promotion logic, and code-size hygiene are the foundation. The mistake would be treating the existing orchestration as permission to launch undisciplined sweeps.

## Work Items
- **P-01 through P-06**: Prerequisites (see tracker.md)
- **X-01 through X-07**: Proxy infrastructure
- **E00**: Baseline P0 smoke
- **E01**: Baseline P1 control
- **E02**: Baseline 8xH100 reproduction
- **E27**: Document-aligned batching — test document-respecting training windows on the published BOS-delimited shards without regenerating data. Implemented by splitting training sequences on `bos_id`, padding short tails with `bos_id`, and masking padded targets with `-100`. Kill if val_bpb regresses by >0.002 or if supervision waste dominates the run. Ref: [NanoGPT speedrun 1.6](../references/nanogpt_speedrun_techniques.md#16-document-aligned-batching). **Complete, killed** — the paired P1 result on current shards was strongly negative.
- **E37**: Skip periodic validation — set `VAL_LOSS_EVERY=0` to remove ~20 validation pauses (~25-30s total) and recover ~900 additional optimizer steps within the same wallclock. Same mechanism as E30 (more steps = better quality). Config-only change, zero code. Kill if Δpq ≥ +0.004. Promote if Δpq ≤ -0.003 or measurably more steps completed.

## Key Principles
- No model conclusion is trustworthy until E02 passes (within 0.003 bpb of baseline); E00 and E01 exist only to validate the harness
- Every run must emit tok_s, train_tokens_seen, prequant exact, and qgap
- Config-driven paths — no hardcoded SP-1024 assumptions
- Budget discipline: phase caps, conservative plan (~12-15 H100-hours)
- All new experiment knobs as env vars (auto-discovered by config_utils)

## Status
Trusted for launch/measurement on both the training path and standalone saved-artifact replay path, with one current caveat: fresh-host `torch.compile` is unstable on Vast right now, so the `E30`, `E34a`, and `E34c` tranches were run under a matched `TORCHDYNAMO_DISABLE=1` eager fallback. `P-01`, `P-02`, `P-03`, `P-04`, `P-05`, `P-06`, `X-01`, `X-02`, `X-05`, `E00`, `E01`, `E02`, `E24a`, `E27`, `E28`, `E30`, `E32`, `E34a`, `E34c`, and `E35` are done, and the first baseline exporter sweeps `E03`/`E04` are also complete and flat. There is no urgent measurement-integrity blocker right now, and the Runpod `1xH100` fallback is now proven for proxy work too. The broader repo next tranche is the first real `8xH100` full-run calibration on Runpod. Detailed closeout review for the Track F experiment branch: [E27](../postmortems/e27_doc_aligned_batching.md).

## Learnings
- MLX lazy eval bug fixed (mx.synchronize → mx.eval)
- 66 infra tests passing
- Codex + Gemini review helped harden the replay utility and narrow the eventual root cause to RoPE cache reconstruction precision
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
- The DIAG proof run `proxy_p1_fast-20260319-204141-603684f1` showed that trainer and replay agree on hyperparameters and loaded-model fingerprints, and that fresh eager replay and fresh compiled replay agree with each other too
- The first deterministic forward mismatch is at `enc0` while `emb` still matches exactly, so the remaining systems blocker is now concentrated inside block 0 runtime state rather than launcher/env/config plumbing
- The next diagnostic tranche is now in the repo: trainer and replay both emit deep block-0 DIAG probes for `attn_norm`, `q/k` before and after RoPE, rotary cache state, `attn_out`, `mlp_out`, and `enc0`, plus cache-cleared and cache-prewarmed variants for the reloaded raw checkpoint path
- `experiments/scripts/compare_diag.py` now provides a stable way to diff `DIAG:*` lines and report the first divergent field, so the next GPU proof run can answer the cache hypothesis directly instead of relying on manual log inspection
- The local regression suite remains green after the bughunt instrumentation tranche: `55` infra tests pass, and the P1 proxy config still validates cleanly
- The proof run `proxy_p1_fast-20260319-230602-1649fc2b` answered the cache hypothesis directly: clearing the rotary caches before `reloaded_prequant_exact` reproduces the standalone replay failure in-process (`1.79098528 / 1.79751456`), while the live cached path remains healthy at `1.40158006 / 1.79692971`
- The block-0 DIAG comparison is now exact enough to name the fault line: trainer `train_reloaded_block0_cache_cleared` and replay `replay_loaded_block0` match field-for-field from `cos/sin` through `enc0`, which means the standalone replay bug is no longer “mysterious fresh-process state”; it is “rebuilding RoPE tables produces a different table than the live cached one”
- The rebuilt rotary table numerics lined up with bf16 reconstruction, while the healthy live cached table was closer to an fp32-built table; the correct systems fix was therefore to make RoPE cache construction precision-stable rather than relying on whatever precision the first cache fill happened to use
- `Rotary` now rebuilds `inv_freq`, `cos`, and `sin` from an fp32 basis and keeps `inv_freq` in fp32 even after `model.bfloat16()`, while returning bf16 views for compute
- Two new regressions now lock that behavior: one proves `inv_freq` stays fp32 after model casting, and one proves clearing/rebuilding the rotary cache does not change the returned tables
- The proof rerun `proxy_p1_fast-20260319-234853-b1623493` closed the systems bug: trainer live `1.40251948/1.40684974`, trainer reloaded `1.40307901/1.40740379`, and fresh-process replay `1.40307901/1.40740286` now agree within noise
- The remaining replay delta is now ordinary eval noise, not a correctness bug: replay-vs-reloaded differs by `+3.6e-10` prequant and `-9.3e-07` postquant, with matching artifact bytes
- The post-fix `E02` replay sanity check `e02_replay_sanity-20260319-190049` verified that replay trust holds on the actual trusted baseline artifact too: replay landed at `1.21973875/1.22687865`, just `+0.00054486/+0.00059058` away from the trusted in-run `E02` metrics
- The paired E27 P1 bundle gave a clean negative result on the published BOS-delimited shards: `phase1_e27_control_p1-20260320-155905-eb77f1a0` landed at postquant `1.41199216`, while `phase1_e27_doc_aligned_p1-20260320-160818-f96fe403` landed at `1.52791342`
- The regression was not subtle: `Δpq=+0.11592126`, `qgap` worsened from `0.00408529` to `0.01193720`, and step time slowed from `416.29 ms` to `640.23 ms` (`+53.8%`)
- The new telemetry explains why this branch failed on current data: only `69.17%` of target positions remained supervised, with `75,972,568` ignored targets by the end of the run
- Systems trust is now restored end to end, and Track F has done its job for this branch too: E27 was cheap to test, decisively negative on the current shard format, and should not stay in the active “likely win” queue unless the data packing approach changes
- Fresh-host compiled warmup is currently broken on Vast for the current environment: both the live trainer and older trusted trainer snapshots fail under `torch.compile`, while the same runs proceed normally under `TORCHDYNAMO_DISABLE=1`
- `E30` therefore used a matched eager fallback and still produced a decisive same-host P1 win: control `phase1_e30_control_p1-20260320-211549-b83629b3` landed at postquant `1.46588267`, while `phase1_e30_batch_schedule_p1-20260320-212550-76454fcd` improved to `1.41664772` (`Δpq=-0.04923495`)
- The systems lesson is to keep the measurement regime explicit: replay trust and artifact trust are still good, but future cheap P1 work on fresh Vast hosts should either use the same eager fallback or prioritize repairing the compiled path before mixing results
- `E34c` also flushed out a real implementation bug quickly: NorMuon's adaptive second-momentum buffer must live in fp32, not bf16, or the optimizer crashes on the first `lerp_` update. With that fixed, the rerun produced a valid slight gain, which means the branch is interpretable rather than tainted by systems noise
