# Track B: Export-Aware Quantization & Exporter Tuning

## Thesis
The 4-hour run proves export retention is the battlefield: only ~40% of pre-quant gains survive roundtrip. The baseline's qgap is 0.0072 bpb; the 4-hour run's is 0.0325. Exporter-only tuning is nearly free and directly targets this. Export-aware training (clamp regularization, QAT-lite) can further close the gap.

## Experiments
- **E03**: Exporter clip-percentile star (sweep INT8_CLIP_PERCENTILE)
- **E04**: Keep-float threshold star (sweep INT8_KEEP_FLOAT_MAX_NUMEL)
- **E13**: Clamp-aware row-outlier regularizer
- **E14**: QAT-lite on selected weights
- **E15**: Best exporter + best export-aware trick composed

## Key Metrics
- qgap (post-roundtrip minus pre-quant bpb) — primary
- Δpq (post-roundtrip delta vs baseline) — ranking
- Compressed model bytes and artifact slack
- Export retention fraction (>0.3 required, >0.5 preferred)
- Payload ratio

## Decision Rules
- E03/E04: promote if any setting improves post-roundtrip by ≥0.002 or saves ≥100kB with ≤0.001 loss
- E13: promote if qgap drops ≥25% or Δpq ≤ -0.003; kill if step time grows >8%
- E14: promote if Δpq ≤ -0.004 and step slowdown <12%

## Status
Not started. Depends on: P-02 (exporter env vars), X-05 (exporter-only eval utility), E02 (baseline reproduction).

## Learnings
(Updated as experiments complete)
