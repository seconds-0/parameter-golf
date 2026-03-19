# Parameter Golf — Master Progress Tracker

> **Read this first on resume.** Full strategy: `docs/experiment_plan_prd.md`. Track details: `docs/tracks/`.

## Prerequisites (complete before experiments)

| Status | ID | Description | Files | Depends |
|--------|-----|-------------|-------|---------|
| [ ] | P-01 | Config-driven data/tokenizer paths (launcher + preflight) | launch_runtime.py, preflight.py | — |
| [ ] | P-02 | Expose exporter controls as env vars (INT8_CLIP_PERCENTILE, etc.) | train_gpt.py | — |
| [ ] | P-03 | Add trainer telemetry (tok_s, train_tokens_seen, prequant exact, qgap) | train_gpt.py | — |
| [ ] | P-04 | Extend parse_log.py (prequant, qgap, artifact slack, payload ratio) | parse_log.py | P-03 |
| [ ] | P-05 | Enrich compare.py and status (qgap, slack, step time, deltas) | compare.py, launch.py | P-04 |
| [ ] | P-06 | Budget discipline: 1xH100 + local machines, reduce budget, phase caps | machines.yaml | — |

## Proxy Infrastructure (complete before Phase 1+)

| Status | ID | Description | Files |
|--------|-----|-------------|-------|
| [ ] | X-01 | P0 smoke proxy config preset | experiments/configs/proxy_p0_smoke.yaml |
| [ ] | X-02 | P1 fast proxy config preset | experiments/configs/proxy_p1_fast.yaml |
| [ ] | X-03 | P2 medium proxy config preset | experiments/configs/proxy_p2_medium.yaml |
| [ ] | X-04 | P3 runtime rehearsal config preset | experiments/configs/proxy_p3_runtime.yaml |
| [ ] | X-05 | Exporter-only eval utility (load final_model.pt, re-export, roundtrip) | experiments/scripts/export_eval.py |
| [ ] | X-06 | Tokenizer stats utility (bytes/token, fragmentation, vocab audit) | experiments/scripts/tokenizer_stats.py |
| [ ] | X-07 | Promotion metadata fields in config_utils (phase, proxy_level, parent_run_id, etc.) | config_utils.py |

## Phase 0: Trust the Measurements

| Status | ID | Description | Proxy | Track | Depends |
|--------|-----|-------------|-------|-------|---------|
| [ ] | E00 | Baseline P0 smoke — harness end-to-end test | P0 | F | P-01..P-06 |
| [ ] | E01 | Baseline P1 control — verify P1 stability for matched deltas | P1 | F | E00 |
| [ ] | E02 | Baseline 8xH100 reproduction — match 1.2244 bpb within 0.003 | Full | F | E01 |

## Phase 1: Free or Nearly Free Wins

| Status | ID | Description | Proxy | Track | Depends |
|--------|-----|-------------|-------|-------|---------|
| [ ] | E03 | Exporter clip-percentile star (sweep INT8_CLIP_PERCENTILE) | Export+P1 | B | E02, X-05 |
| [ ] | E04 | Keep-float threshold star (sweep INT8_KEEP_FLOAT_MAX_NUMEL) | Export+P1 | B | E02, X-05 |

## Phase 2: Tokenizer First, Then Recipe

| Status | ID | Description | Proxy | Track | Depends |
|--------|-----|-------------|-------|-------|---------|
| [ ] | E05 | Tokenizer audit bundle (offline economics analysis) | Offline | A | X-06 |
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

- P1 promote threshold: Δpq ≤ -0.003, or qgap ≤ -20%, or clear throughput/artifact win
- P2 promote threshold: Δpq ≤ -0.005, or Δpq ≤ -0.003 with strong side benefit
- 8xH100 gate: 2-seed mean Δpq ≤ -0.006, no seed worse than baseline, qgap not worse by >0.002
- Kill: worse than baseline by ≥0.004 at P1, or ≥0.005 at P2 → immediate kill

## Budget

- Conservative plan: ~12-15 H100-hours total
- Phase caps: P0 10%, P1 25%, P2 35%, P3 20%, P4 10% reserved
- Current spend: $0.00 (no runs yet)
