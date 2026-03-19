# Track B: Export-Aware Quantization & Exporter Tuning

## Thesis
The 4-hour run proves export retention is the battlefield: only ~40% of pre-quant gains survive roundtrip. The baseline's qgap is 0.0072 bpb; the 4-hour run's is 0.0325. Exporter-only tuning is nearly free and directly targets this. Export-aware training (clamp regularization, QAT-lite) can further close the gap.

## Experiments
- **E03**: Exporter clip-percentile star (sweep INT8_CLIP_PERCENTILE)
- **E04**: Keep-float threshold star (sweep INT8_KEEP_FLOAT_MAX_NUMEL)
- **E13**: Clamp-aware row-outlier regularizer
- **E14**: QAT-lite on selected weights
- **E15**: Best exporter + best export-aware trick composed
- **E24**: Weight decay / L2 penalty for export retention (sweep 0.1–1.6) — inspired by [Q Labs 10x blog](../references/qlabs_10x_data_efficiency.md). **Unblocked** — depends only on E02, uses in-process roundtrip (not X-05). Kill if pre-quant val_bpb regresses by >0.005 without compensating qgap improvement.

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
Blocked on fresh-process replay correctness. `P-02` is complete and `E02` passed on `baseline_repro-20260319-093041-82dade88`, but fresh 1xH100 replays of saved `final_model.pt` artifacts still do not reproduce the trusted in-run metrics. Multiple fresh instrumented runs proved the in-trainer reload path is healthy, and persisting RoPE `inv_freq` did not fix standalone replay, so `E03` and `E04` remain paused until the fresh-process reconstruction path is explained.

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
