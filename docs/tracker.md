# Parameter Golf — Master Progress Tracker

> **Read this first on resume.** Full strategy: `docs/experiment_plan_prd.md`. Track details: `docs/tracks/`.

## Prerequisites (complete before experiments)

| Status | ID | Description | Files | Depends |
|--------|-----|-------------|-------|---------|
| [x] | P-01 | Config-driven data/tokenizer paths (launcher + preflight) | launch_runtime.py, preflight.py | — |
| [x] | P-02 | Expose exporter controls as env vars (INT8_CLIP_PERCENTILE, etc.) | train_gpt.py | — |
| [x] | P-03 | Add trainer telemetry (tok_s, train_tokens_seen, prequant exact, qgap) | train_gpt.py | — |
| [x] | P-04 | Extend parse_log.py (prequant, qgap, artifact slack, payload ratio) | parse_log.py | P-03 |
| [x] | P-05 | Enrich compare.py and status (qgap, slack, step time, deltas) | compare.py, launch.py | P-04 |
| [x] | P-06 | Budget discipline: 1xH100 + local machines, reduce budget, phase caps | machines.yaml | — |

## Proxy Infrastructure (complete before Phase 1+)

| Status | ID | Description | Files |
|--------|-----|-------------|-------|
| [x] | X-01 | P0 smoke proxy config preset | experiments/configs/proxy_p0_smoke.yaml |
| [x] | X-02 | P1 fast proxy config preset | experiments/configs/proxy_p1_fast.yaml |
| [ ] | X-03 | P2 medium proxy config preset | experiments/configs/proxy_p2_medium.yaml |
| [ ] | X-04 | P3 runtime rehearsal config preset | experiments/configs/proxy_p3_runtime.yaml |
| [ ] | X-05 | Exporter-only eval utility (load final_model.pt, re-export, roundtrip) | experiments/scripts/export_eval.py |
| [ ] | X-06 | Tokenizer stats utility (bytes/token, fragmentation, vocab audit) | experiments/scripts/tokenizer_stats.py |
| [ ] | X-07 | Promotion metadata fields in config_utils (phase, proxy_level, parent_run_id, etc.) | config_utils.py |

## Phase 0: Trust the Measurements

| Status | ID | Description | Proxy | Track | Depends |
|--------|-----|-------------|-------|-------|---------|
| [x] | E00 | Baseline P0 smoke — harness end-to-end run+parse check, not a research gate | P0 | F | P-01, P-03, P-04, X-01 |
| [x] | E01 | Baseline P1 control — verify P1 stability for matched deltas | P1 | F | E00, X-02 |
| [x] | E02 | Baseline 8xH100 reproduction — match 1.2244 bpb within 0.003 | Full | F | E01, P-06 |

## Phase 1: Free or Nearly Free Wins

| Status | ID | Description | Proxy | Track | Depends |
|--------|-----|-------------|-------|-------|---------|
| [ ] | E03 | Exporter clip-percentile star (sweep INT8_CLIP_PERCENTILE) | Export+P1 | B | E02, P-02, X-05 |
| [ ] | E04 | Keep-float threshold star (sweep INT8_KEEP_FLOAT_MAX_NUMEL) | Export+P1 | B | E02, P-02, X-05 |

## Phase 2: Tokenizer First, Then Recipe

| Status | ID | Description | Proxy | Track | Depends |
|--------|-----|-------------|-------|-------|---------|
| [ ] | E05 | Tokenizer audit bundle (offline economics analysis) | Offline | A | E02, P-01, X-06 |
| [ ] | E06 | SP-512 P1 | P1 | A | E05 |
| [ ] | E07 | SP-768 P1 | P1 | A | E05 |
| [ ] | E08 | SP-1536 P1 | P1 | A | E05 |
| [ ] | E09 | Best tokenizer P2 (winner from E06-E08) | P2 | A | E06..E08 |
| [ ] | E10 | Tied-embed LR star on tokenizer winner | P1 | C | E09 |
| [ ] | E11 | Matrix/scalar LR star on tokenizer winner | P1 | C | E09 |
| [ ] | E12 | Embedding norm penalty A/B | P1→P2 | C | E09 |

## Phase 3: Export-Aware Training

| Status | ID | Description | Proxy | Track | Depends |
|--------|-----|-------------|-------|-------|---------|
| [ ] | E13 | Clamp-aware row-outlier regularizer | P1→P2 | B | E03..E04 |
| [ ] | E14 | QAT-lite on selected weights | P1 | B | E03..E04 |
| [ ] | E15 | Best exporter + best export-aware trick composed | P2 | B | E13 or E14 |

## Phase 4: Byte-Efficient Capacity Trades

| Status | ID | Description | Proxy | Track | Depends |
|--------|-----|-------------|-------|-------|---------|
| [ ] | E16 | KV-head rebudget (4→2) | P1+P3 | D | E02 |
| [ ] | E17 | KV-head + width rebudget | P1→P3→P2 | D | E16 |
| [ ] | E18 | Grouped layer sharing moonshot | P1 | E | E02 |

## Phase 5: Composition and Promotion

| Status | ID | Description | Proxy | Track | Depends |
|--------|-----|-------------|-------|-------|---------|
| [ ] | E19 | Composed finalist P2, seed A | P2 | — | Best of E09..E15 |
| [ ] | E20 | Composed finalist P2, seed B | P2 | — | E19 |
| [ ] | E21 | 8xH100 runtime rehearsal | P3 | — | E20 |
| [ ] | E22 | Full 8xH100 candidate run | Full | — | E21 |

## Key Metrics (always track these)

- **Δpq**: candidate post-roundtrip val_bpb minus baseline post-roundtrip val_bpb (negative = better)
- **qgap**: post-roundtrip val_bpb minus pre-quant val_bpb (smaller = better)
- **Artifact slack**: 16,000,000 minus total submission bytes
- **Export retention**: post-roundtrip gain / pre-quant gain (>0.3 required, >0.5 preferred)

## Decision Rules

- Rank bundles by post-roundtrip Δpq first, then qgap, then wallclock headroom, then artifact slack
- P1 promote threshold: Δpq ≤ -0.003, or qgap reduced by at least 20% vs matched baseline with neutral quality, or clear throughput/artifact win
- P2 promote threshold: Δpq ≤ -0.005, or Δpq ≤ -0.003 with strong side benefit
- 8xH100 gate: 2-seed mean Δpq ≤ -0.006, no seed worse than baseline, qgap not worse by >0.002
- Kill: worse than baseline by ≥0.004 at P1, or ≥0.005 at P2 → immediate kill

## Budget

- Conservative plan: ~12-15 H100-hours total
- Phase caps: P0 10%, P1 25%, P2 35%, P3 20%, P4 10% reserved
- Current spend: ~$6.20

## Current State

- `E00` passed on `proxy_p0_smoke-20260319-082117-62aaaea6`
- `E00` reference metrics: postquant `val_bpb=2.59277612`, prequant `val_bpb=2.58420195`, `qgap_bpb=0.00857417`
- `E00` artifact result: `total_submission_bytes=6,649,600`, `artifact_slack_bytes=9,350,400`
- `E00` throughput result: `tok_s=1.58M`, `train_tokens_seen=71,303,168`, `step_avg_ms=331.47`
- First failed smoke exposed a launcher bug: `run` was defaulting to `8` GPUs on a `1xH100` target
- Second failed smoke exposed a missing Triton toolchain precondition (`python3.11-dev`) and an overly aggressive watchdog stall rule
- Exporter controls now exposed: `INT8_CLIP_PERCENTILE`, `INT8_KEEP_FLOAT_MAX_NUMEL`, `CONTROL_TENSOR_NAME_PATTERNS`, and `INT8_KEEP_FLOAT_FP32_NAME_PATTERNS`
- Exporter-only exact roundtrip utility now exists at `experiments/scripts/export_eval.py`
- `E01` passed on the successful seed pair `phase0_e01_seed1337-20260319-085619-0bcbed04` and `phase0_e01_seed2024-20260319-090404-51ce3bc4`
- `E01` mean postquant `val_bpb=1.38512509`, spread `0.00278616`, mean prequant `val_bpb=1.38263008`, mean `qgap_bpb=0.00249501`
- `E01` throughput/slack baseline: `step_avg_ms≈336.8`, `tok_s≈1.56M`, `artifact_slack_bytes≈3.99M`
- The first `E01` attempts failed because a fresh Vast host came up on `Python 3.11.0rc1`; switching the box to `Python 3.11.15` fixed TorchInductor stability
- Fresh `8xH100` bring-up exposed two more harness issues that are now fixed: preflight/runtime now prefer the shared repo `.venv/bin/python` on remote hosts, and `data/cached_challenge_fineweb.py` now defaults to the full published train split instead of silently stopping at `80` train shards
- `E02` passed on `baseline_repro-20260319-093041-82dade88`
- `E02` reference metrics: postquant `val_bpb=1.22628807`, prequant `val_bpb=1.21919389`, `qgap_bpb=0.00709417`
- `E02` artifact/runtime result: `total_submission_bytes=15,861,240`, `artifact_slack_bytes=138,760`, `step_avg_ms=43.55`, `tok_s=12.04M`, `train_tokens_seen=7,224,164,352`
- `E02` clears the reproduction gate: it is `+0.00192237` vs the published `1.22436570`, with artifact size only `2,249` bytes smaller than the published `15,863,489`
- `export_eval.py` is now stricter and safer: it can replay from a checkpoint's neighboring `manifest.json` + `train_gpt.py` snapshot, it scrubs ambient trainer env vars before import, and it no longer overwrites `final_model.int8.ptz` by default
- The `X-05` blocker is deeper than config/env drift. A fresh 1xH100 replay of the trusted `E02` checkpoint still produced `prequant_val_bpb=1.81346210` and `postquant_val_bpb=1.82621440` even when using the run's own manifest env and trainer snapshot
- The same raw-checkpoint replay failure shows up on the 1xH100 `E01` control artifact: `phase0_e01_seed1337-20260319-085619-0bcbed04/final_model.pt` replayed at `prequant_val_bpb=1.79180195` and `postquant_val_bpb=1.79691088` instead of the trusted in-run `1.38394218/1.38651817`
- That means `X-05` is not just an exporter utility bug: the saved `final_model.pt` artifacts do not currently reproduce the live end-of-run metrics, so Track B stays blocked until the trainer-side serialization path is debugged
- The trainer now emits replay-trust proof signals at the end of a run: `uncompiled_check`, `checkpoint_save_verify`, `reloaded_prequant_exact`, and `reloaded_int8_zlib_roundtrip_exact`
- `parse_log.py`, `compare.py`, and `launch.py status` now surface those replay-trust signals directly, so the next real run can tell us whether the gap is compiled-vs-eager, save/load drift, or both
- A second fresh instrumented P1 proof run succeeded on `proxy_p1_fast-20260319-192237-39db664f`: in-trainer replay trust was again good, with `uncompiled_check_delta_bpb=+0.00056740`, `checkpoint_save_verify_max_abs_diff=0.0`, `reloaded_prequant_delta_bpb=+0.00056740`, and `reloaded_postquant_delta_bpb=+0.00057580`
- The checkpoint now explicitly serializes RoPE `inv_freq`, so the remaining mismatch is no longer explainable as “forward-relevant buffers were missing from state_dict`
- Even on the original remote artifact, standalone fresh-process replay is still badly wrong: `final_model.export_eval.json` landed at `prequant_val_bpb=1.78922077` and `postquant_val_bpb=1.79489698` instead of the trainer-side `1.40549047/1.40989139`
- The next work in sequence is now a focused fresh-process reconstruction bughunt inside `export_eval.py` and the trainer snapshot import path: explain why a fresh process built from the same checkpoint, config, tokenizer path, and dataset path still diverges while the in-process trainer reload stays within about `0.0006`
