# Parameter Golf — Master Progress Tracker

> **Read this first on resume.** Full strategy: `docs/experiment_plan_prd.md`. Track details: `docs/tracks/`. Composition rules: `docs/experiment_interactions.md`.

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
| [x] | X-05 | Exporter-only eval utility (load final_model.pt, re-export, roundtrip) | experiments/scripts/export_eval.py |
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
| [x] | E03 | Exporter clip-percentile star (sweep INT8_CLIP_PERCENTILE) | Export+P1 | B | E02, P-02, X-05 |
| [x] | E04 | Keep-float threshold star (sweep INT8_KEEP_FLOAT_MAX_NUMEL) | Export+P1 | B | E02, P-02, X-05 |
| [x] | E23 | EMA weight averaging at export (decay sweep: 0.999, 0.9999) | P1 | C | E02 |
| [x] | E27 | Document-aligned batching on published BOS-delimited shards | P1 | F | E02 |
| [ ] | E28 | Asymmetric softcap: separate cap_pos/cap_neg, test (30,20),(30,15),(20,30) | P1 | C | E02 |
| [x] | E32 | WSD LR schedule (Warmup-Stable-Decay replacing baseline warmdown) | P1 | C | E02 |
| [ ] | E35 | Higher β₂ during cooldown (0.97-0.99 in decay phase) | P1 | C | E02 |

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
| [ ] | E29 | Value embed gate: per-block scalar ve_gate_w (init=0), V += gate*tok_emb[:kv_dim] | P1 | E | E02 |
| [ ] | E30 | Batch schedule: 131K tokens first 30% of steps, then 524K for remaining 70% | P1 | C | E02 |
| [ ] | E34a | Polar Express drop-in (optimal polynomial NS replacement, ~50 lines) | P1 | C | E02 |
| [ ] | E34b | Turbo-Muon AOL preconditioning (if E34a insufficient, ~200 lines) | P1 | C | E34a |
| [ ] | E34c | NorMuon neuron-wise adaptive LR (~150 lines) | P1 | C | E02 |

## Phase 3: Export-Aware Training

| Status | ID | Description | Proxy | Track | Depends |
|--------|-----|-------------|-------|-------|---------|
| [ ] | E13 | Clamp-aware regularizer: λ * mean(max(0, \|W\|-clip)²), sweep λ ∈ {0.001,0.01,0.1} | P1→P2 | B | E03..E04 |
| [ ] | E15 | Compose best exporter settings + best regularizer winner (composition) | P2 | B | E24,E33,E13 |
| [ ] | E24 | Weight decay for export retention (cautious gated + fixed, sweep 0.1–1.6) | P1→P2 | B | E02 |
| [ ] | E33 | Range Reg R²: λ * mean(max-min per row), sweep λ ∈ {0.001,0.01,0.1} | P1→P2 | B | E02 |
| [ ] | E14b | Quant noise injection: σ(t)=0.5*exp(-0.01t), delayed step 2000, ~2% overhead | P1 | B | E02 |
| [ ] | E14a | STE fake-quantize all 2D matrices, delayed step 1000, ~10% overhead | P1 | B | E02 |
| [ ] | E14c | Learned per-row scales (LSQ-lite), own LR, ~8% overhead | P1 | B | E14b or E14a |

## Phase 4: Byte-Efficient Capacity Trades

| Status | ID | Description | Proxy | Track | Depends |
|--------|-----|-------------|-------|-------|---------|
| [ ] | E16 | KV-head rebudget (4→2) | P1+P3 | D | E02 |
| [ ] | E17 | KV-head + width rebudget (composition: reinvest E16 savings) | P1→P3→P2 | D | E16 |
| [ ] | E18 | Layer sharing (6/5/4 unique layers looped to 9-12 effective) | P1 | E | E02 |
| [ ] | E25 | SwiGLU activation replacing ReLU² (matched param budget) | P1 | E | E02 |
| [ ] | E26 | Layer sharing + SwiGLU rebudget (composition: E18 savings → SwiGLU hidden=704) | P1→P2 | E | E18, E25 |
| [ ] | E31a | MTP: 1 aux head (t+2), equal weight, no curriculum, ~8-12% overhead | P1 | E | E02 |
| [ ] | E31b | MTP: 2 aux heads (t+2,t+3), forward curriculum phased in, ~15-20% overhead | P1 | E | E31a |
| [ ] | E31c | MTP: decaying loss weights (λ_k = 0.5^(k-1)) on E31b architecture | P1 | E | E31b |

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
- Current spend: ~$9.50

## Ideas to Explore (not yet precise enough for a kill-rule experiment)

(Empty — E14, E31, E34 were decomposed into sub-experiments E14a/b/c, E31a/b/c, E34a/b/c and promoted back to the phase tables.)

## Current State

- Replay and training trust are restored end to end. `E02` reproduced the published baseline on `baseline_repro-20260319-093041-82dade88` at postquant `val_bpb=1.22628807`, prequant `val_bpb=1.21919389`, `qgap_bpb=0.00709417`, `total_submission_bytes=15,861,240`, and `step_avg_ms=43.55`.
- `X-05` is complete and trusted. The post-fix replay sanity bundle `e02_replay_sanity-20260319-190049` replayed the trusted `E02` checkpoint at prequant `1.21973875` and postquant `1.22687865`, only `+0.00054486 / +0.00059058` away from the in-run `E02` metrics.
- `E03` and `E04` are complete and flat on the trusted baseline checkpoint. `e03_clip_star-20260319-190645` found no clip-percentile win, and `e04_keep_float_star-20260319-191446` found no keep-float threshold win worth promoting under the byte cap.
- `E27` is now complete and killed on the current published BOS-delimited shards. The paired same-host P1 bundle `phase1_e27_control_p1-20260320-155905-eb77f1a0` vs `phase1_e27_doc_aligned_p1-20260320-160818-f96fe403` regressed post-roundtrip from `1.41199216` to `1.52791342` (`Δpq=+0.11592126`), worsened `qgap` from `0.00408529` to `0.01193720`, and slowed step time from `416.29 ms` to `640.23 ms` (`+53.8%`). The loader only supervised `69.17%` of target positions on the published shards, so this is not an active baseline modifier.
- `E23` is now complete and killed. The matched same-host P1 control `phase1_e23_control_p1-20260320-163308-38d0b485` landed at postquant `1.38639890` with `qgap=0.00272267`. The EMA export candidates were dramatically worse despite healthy live training metrics: `phase1_e23_ema0999_p1-20260320-164244-1b5abd14` landed at `2.03365982` (`Δpq=+0.64726092`, `qgap=0.00562494`), and `phase1_e23_ema09999_p1-20260320-165137-f8faa3a1` landed at `5.45891420` (`Δpq=+4.07251530`, `qgap=0.03394086`). Both live prequant checks stayed near baseline before the EMA swap, so this branch failed specifically at the export-smoothed snapshot, not in the training path.
- `E32` is now complete and promoted. The matched same-host P1 control `phase1_e32_control_p1-20260320-172313-5aaa1c50` landed at postquant `1.46474630`, prequant `1.45614906`, and `qgap=0.00859724` with `step_avg_ms=557.65`. The WSD candidate `phase1_e32_wsd_p1-20260320-173257-a73c389c` improved to postquant `1.44033240`, prequant `1.43925826`, and `qgap=0.00107415`, for `Δpq=-0.02441390`, `Δpre=-0.01689080`, and `Δqgap=-0.00752309`, while step time only rose to `568.16 ms` (`+1.88%`). This is a clear P1 promotion and makes WSD the active base schedule for the next recipe follow-up.
- Static exporter-only tuning is therefore exhausted for this checkpoint. If we stay in Track B, the next concrete branch is `E24` weight decay for export retention, followed later by `E33` or `E13` one at a time.
- The best overall next cheap tranche is now `E35` on top of the promoted WSD base, then `E28` asymmetric logit rescale. If we choose to push Track B next instead, `E24` is the right branch.
- Tokenizer work is still blocked on `X-06` then `E05`, and tokenizer-dependent recipe stars `E10`-`E12` remain blocked on `E09`.

## Recent Learnings

- The early harness failures were worth fixing before research: smoke runs exposed the wrong-GPU-count default, a missing Triton toolchain precondition, an over-eager watchdog, and a remote interpreter mismatch between bare `python3` and the shared repo `.venv`.
- The replay bug was model-side, not exporter-side. Fresh-process replay only became trustworthy after making RoPE cache rebuilds precision-stable in fp32 and proving the fix with trainer-side reload checks plus block-0 DIAG comparisons.
- Exporter controls are now fully harnessed via env vars, replay is safe and reproducible, and the exporter-only stars established a useful negative result: clip percentile and keep-float threshold are not where the next meaningful win is hiding for this baseline checkpoint.
- `E27` was worth testing, but the current shard format matters: using BOS-delimited document alignment without regenerating data or adding a denser packing scheme left roughly `31%` of target positions ignored, which overwhelmed any hoped-for boundary-respect gain and makes this branch a clear kill on today's published dataset.
- `E23` was also worth testing because it cleanly separated live training quality from exported-model quality. On this short P1 proxy, both EMA decays left the live prequant path near baseline but made the exported checkpoint catastrophically worse, which is a strong sign that naive long-horizon export-only EMA is the wrong fit for this regime as implemented.
- `E32` is the first strong positive result in the cheap independent tranche: WSD did not just shave `qgap`; it improved both prequant and post-roundtrip quality materially on a matched same-host P1 run, which is exactly the profile we want before layering anything on top.
