# Track B: Export-Aware Quantization & Exporter Tuning

## Thesis
The 4-hour run proves export retention is the battlefield: only ~40% of pre-quant gains survive roundtrip. The baseline's qgap is 0.0072 bpb; the 4-hour run's is 0.0325. Exporter-only tuning is nearly free and directly targets this. Export-aware training (clamp regularization, QAT-lite) can further close the gap.

## Experiments
- **E03**: Exporter clip-percentile star (sweep INT8_CLIP_PERCENTILE)
- **E04**: Keep-float threshold star (sweep INT8_KEEP_FLOAT_MAX_NUMEL)
- **E12**: Embedding norm penalty: A=L2 λ=0.01, B=max-norm clip to 95th pct. Kill if val_bpb regresses >0.003 without compensating qgap improvement.
- **E13**: Clamp-aware regularizer — add `λ * mean(max(0, |W_ij| - clip_threshold_i)²)` per 2D weight matrix to the training loss, where `clip_threshold_i` is the per-row int8 clip boundary. Sweep λ ∈ {0.001, 0.01, 0.1}. Kill if Δpq ≥ +0.004 or step time >8%.
- **E15**: Compose best exporter settings + best regularizer winner (composition step). Gate: run only after E24/E33/E13 have P1 results.
- **E24a**: Fixed L2 weight decay, sweep {0.1, 0.5, 1.0} — standard L2 weight decay on 2D matrices, from [Q Labs](../references/qlabs_10x_data_efficiency.md). **Unblocked** — depends only on E02, uses in-process roundtrip (not X-05). Kill if pre-quant val_bpb regresses by >0.005 without compensating qgap improvement.
- **E24b**: Cautious gated decay, sweep {0.1, 0.5, 1.0} — only applies decay when weight and gradient are aligned, from [NanoGPT speedrun record 44](../references/nanogpt_speedrun_techniques.md#12-cautious-weight-decay). **Unblocked** — depends only on E02, uses in-process roundtrip (not X-05). Kill if pre-quant val_bpb regresses by >0.005 without compensating qgap improvement.
- **E33**: Range Regularization R² — add `λ * mean_over_rows(max(W_row) - min(W_row))` per 2D weight matrix to the training loss. Distinct from E24 (targets magnitude) and E13 (targets row outliers). R² targets the distribution range directly since int8 scale = range/255. Sweep λ ∈ {0.001, 0.01, 0.1}. Kill if pre-quant regresses >0.004 without qgap improvement ≥20%. Ref: [small model landscape](../references/small_model_optimization_landscape.md#2-range-regularization-r-for-quantization). **Unblocked** — depends only on E02.
- **E14b**: Quantization noise injection — add `W + σ(t) * randn_like(W)` in forward pass where `σ(t) = 0.5 * exp(-0.01 * t)`. Delayed activation at step 2000. ~2% step overhead, ~15 lines. Kill if Δpq ≥ +0.004. **Run first** (cheapest QAT variant).
- **E14a**: STE fake-quantize — simulate per-row int8 quantize-dequantize on all 2D weight matrices using straight-through estimator. Delayed activation at step 1000. ~10% step overhead, ~20 lines. Kill if Δpq ≥ +0.004 or step time >12%.
- **E14c**: Learned per-row scales (LSQ-lite) — replace fixed 99.99984th percentile clip with one learnable scale scalar per row, trained at low LR. ~8% step overhead, ~40 lines. Kill if Δpq ≥ +0.004. Run only if E14b or E14a shows promise.

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
Unblocked, but the static exporter-only stars are now complete and flat. `P-02` is complete, `E02` passed on `baseline_repro-20260319-093041-82dade88`, and `X-05` is trusted on both a fresh P1 proof run and the actual trusted `E02` baseline checkpoint. `E03` and `E04` both finished with `no win`, and `E23` export-time EMA has now also been tested and killed, so the next worthwhile Track B step is `E24a` rather than more exporter-only knob sweeps or more export-smoothing exploration. The broader repo has now also closed out `E34c` as a below-threshold near-miss, so `E24a` is no longer just the next Track B branch; it is the live next diversification move. Detailed closeout reviews: [E03](../postmortems/e03_exporter_clip_star.md), [E04](../postmortems/e04_keep_float_threshold.md).

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
- The unblocker was a model-side RoPE precision bug, not an exporter bug: rebuilding rotary tables in bf16 changed the attention path enough to ruin replay correctness, even though saved weights and config matched
- `Rotary` now rebuilds its basis and cache tables in fp32, keeps `inv_freq` in fp32 across model casting, and only casts returned `cos/sin` views down to the compute dtype
- The proof rerun `proxy_p1_fast-20260319-234853-b1623493` shows the replay path is now trustworthy again: trainer live `1.40251948/1.40684974`, trainer reloaded `1.40307901/1.40740379`, and fresh-process replay `1.40307901/1.40740286`
- Replay-vs-reloaded agreement is effectively exact on that run: prequant delta `+3.6e-10`, postquant delta `-9.3e-07`, and `total_submission_bytes=11,290,179` matches exactly
- Block-0 DIAGs now agree through the actual computation path after cache rebuild: trainer `train_reloaded_block0_cache_cleared` and replay `replay_loaded_block0` differ only in whether the cache was already populated before the probe, not in any of the forward values
- The post-fix `E02` replay sanity bundle `e02_replay_sanity-20260319-190049` confirms the fix on the real baseline artifact too: replay landed at prequant `1.21973875`, postquant `1.22687865`, `qgap_bpb=0.00713990`, and `total_submission_bytes=15,874,097`
- Those `E02` replay deltas are comfortably inside the acceptance band: `+0.00054486` prequant and `+0.00059058` postquant versus the trusted in-run `E02` values, so the exporter-only path is now trusted on the exact checkpoint `E03` and `E04` will use
- `E03` finished on bundle `e03_clip_star-20260319-190645` and came back flat: the nominal winner was `INT8_CLIP_PERCENTILE=99.99998` at postquant `1.22687860`, but that is still `+0.00059053` worse than the trusted `E02` baseline and therefore not promote-worthy
- The default clip `99.99984` was effectively tied with the winner at `1.22687865`, and the whole star stayed in a tight but consistently worse band from `+0.00059053` to `+0.00067147` versus `E02`
- `E04` then kept `INT8_CLIP_PERCENTILE=99.99984` fixed and swept `INT8_KEEP_FLOAT_MAX_NUMEL` on bundle `e04_keep_float_star-20260319-191446`
- The `E04` result was also flat: `8192`, `16384`, `32768`, and `65536` all landed at exactly `1.22687865` postquant with identical bytes, while `131072` improved to `1.22675124` only by blowing the byte cap at `18,054,259` total bytes
- The exporter-only conclusion is now clear for this checkpoint: these two static knobs are not where the meaningful win is hiding, so more clip/threshold stars would be wheel-spinning
- If Track B is the next lane we want to push, `E24a` fixed L2 weight decay is the next concrete candidate rather than more exporter-only knob sweeps
