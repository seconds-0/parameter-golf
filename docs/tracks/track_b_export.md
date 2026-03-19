# Track B: Export-Aware Quantization & Exporter Tuning

## Thesis
The 4-hour run proves export retention is the battlefield: only ~40% of pre-quant gains survive roundtrip. The baseline's qgap is 0.0072 bpb; the 4-hour run's is 0.0325. Exporter-only tuning is nearly free and directly targets this. Export-aware training (clamp regularization, QAT-lite) can further close the gap.

## Experiments
- **E03**: Exporter clip-percentile star (sweep INT8_CLIP_PERCENTILE)
- **E04**: Keep-float threshold star (sweep INT8_KEEP_FLOAT_MAX_NUMEL)
- **E13**: Clamp-aware row-outlier regularizer
- **E14**: QAT-lite on selected weights
- **E15**: Best exporter + best export-aware trick composed
- **E24**: Weight decay for export retention — test both **fixed L2** (sweep 0.1–1.6, from [Q Labs](../references/qlabs_10x_data_efficiency.md)) and **cautious gated decay** (from [NanoGPT speedrun record 44](../references/nanogpt_speedrun_techniques.md#12-cautious-weight-decay)). Cautious variant only applies decay when weight and gradient are aligned, avoiding counterproductive updates. **Unblocked** — depends only on E02, uses in-process roundtrip (not X-05). Kill if pre-quant val_bpb regresses by >0.005 without compensating qgap improvement.
- **E33**: Range Regularization R² — penalizes the range (max-min) of weight values per row/tensor during training, tightening distributions for better int8 quantization. Distinct from E24 (targets magnitude) and E13 (targets row outliers). R² targets the distribution shape itself. Optionally bundle with **KURE** (kurtosis regularization, arXiv:2602.03614) which penalizes spiky distributions. Ref: [small model landscape](../references/small_model_optimization_landscape.md#2-range-regularization-r-for-quantization). **Unblocked** — depends only on E02. Kill if pre-quant regresses >0.004 without qgap improvement.

## Key Metrics
- Δpq (post-roundtrip delta vs baseline) — primary ranking metric
- qgap (post-roundtrip minus pre-quant bpb) — retention diagnostic / tie-breaker
- Compressed model bytes and artifact slack
- Export retention fraction (>0.3 required, >0.5 preferred)
- Payload ratio

## Decision Rules
- Rank exporter settings by post-roundtrip Δpq first; use qgap to break ties or explain wins/losses
- E03/E04: promote if any setting improves post-roundtrip by ≥0.002 or saves ≥100kB with ≤0.001 loss
- E13: promote if qgap drops ≥25% or Δpq ≤ -0.003; kill if step time grows >8%
- E14: promote if Δpq ≤ -0.004 and step slowdown <12%

## Status
Blocked on fresh-process replay correctness. `P-02` is complete and `E02` passed on `baseline_repro-20260319-093041-82dade88`, but fresh 1xH100 replays of saved `final_model.pt` artifacts still do not reproduce the trusted in-run metrics. Multiple fresh instrumented runs proved the in-trainer reload path is healthy, and persisting RoPE `inv_freq` did not fix standalone replay, so `E03` and `E04` remain paused until the fresh-process reconstruction path is explained by the new block-0 cache probes.

## Learnings
- Exporter controls currently exposed via env vars: `INT8_CLIP_PERCENTILE`, `INT8_KEEP_FLOAT_MAX_NUMEL`, `CONTROL_TENSOR_NAME_PATTERNS`, and `INT8_KEEP_FLOAT_FP32_NAME_PATTERNS`
- `export_eval.py` now prefers the checkpoint's neighboring `manifest.json` + `train_gpt.py` snapshot when available, scrubs ambient trainer env vars before import, and writes debug artifacts to `final_model.export_eval.int8.ptz` so it does not clobber a trusted `final_model.int8.ptz`
- The trusted reference checkpoint is `baseline_repro-20260319-093041-82dade88/final_model.pt`, with exact prequant `val_bpb=1.21919389`, postquant `val_bpb=1.22628807`, `qgap_bpb=0.00709417`, and `total_submission_bytes=15,861,240`
- A stricter default-setting replay of that same `E02` checkpoint on 1xH100 still failed badly: `x05_replay_check.json` came back at `prequant_val_bpb=1.81346210` and `postquant_val_bpb=1.82621440`, which means the blocker is not just config drift inside `export_eval.py`
- The same failure appears on the 1xH100 `E01` control artifact: `x05_e01_replay_check.json` replayed `phase0_e01_seed1337-20260319-085619-0bcbed04/final_model.pt` at `1.79180195/1.79691088` instead of the trusted in-run `1.38394218/1.38651817`
- The trainer now emits `uncompiled_check`, `checkpoint_save_verify`, `reloaded_prequant_exact`, and `reloaded_int8_zlib_roundtrip_exact`, and the parser/status surfaces carry those through into metrics and run summaries
- A fresh instrumented proof run on `proxy_p1_fast-20260319-185044-b42cf8f0` showed the trainer-side replay path is fine: `uncompiled_check_delta_bpb=+0.00056196`, `checkpoint_save_verify_max_abs_diff=0.0`, `reloaded_prequant_delta_bpb=+0.00056196`, and `reloaded_postquant_delta_bpb=+0.00057997`
- But the standalone replay path is still broken on that same fresh checkpoint: `final_model.export_eval.json` replayed it at `1.78986763/1.79694755`, so the remaining blocker is now specifically a fresh-process reconstruction/export-eval bug rather than a raw checkpoint save bug
- A second fresh proof run on `proxy_p1_fast-20260319-192237-39db664f` reproduced the same shape of failure directly on the original remote artifact: trainer-side reload stayed tight at `1.40549047/1.40989139`, while standalone replay still came back at `1.78922077/1.79489698`
- RoPE `inv_freq` is now persisted in `final_model.pt`, and that still did not move the fresh-process replay failure, so the next suspect is not “missing RoPE basis in state_dict”; it is a deeper fresh-process model reconstruction or snapshot-import mismatch
- The DIAG run `proxy_p1_fast-20260319-204141-603684f1` tightened that further: replay hyperparameters and loaded-model fingerprints match the trainer-side diagnostics exactly, and the first deterministic forward mismatch appears at `enc0`
- Fresh eager replay and fresh compiled replay also match each other exactly on that same checkpoint, so Track B is specifically blocked on hidden same-process runtime state from the trained module, not on missing replay config or missing replay compilation
- Replay tooling now emits deep block-0 DIAG lines for the raw checkpoint path, including cache-cleared and cache-prewarmed variants, and `experiments/scripts/compare_diag.py` can diff them against trainer logs to isolate the first divergent field automatically
- The proof run `proxy_p1_fast-20260319-230602-1649fc2b` answered that question: clearing the trainer-side rotary caches makes the in-process raw-checkpoint path fall directly into the standalone replay regime, and replay matches the cache-cleared trainer block-0 payload exactly
- Track B is therefore blocked on one concrete model-side defect, not on the exporter utility: RoPE cache reconstruction is precision-sensitive, and rebuilt tables differ enough from the live cached tables to move prequant `val_bpb` from `1.40158006` to `1.79098528`
- The next unblocker is now a Rotary fix plus one proof rerun: build RoPE caches from a stable fp32 basis, re-verify that trainer reload and fresh-process replay both return to the healthy prequant path, then reopen `E03` and `E04`
