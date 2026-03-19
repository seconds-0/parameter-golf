# Track C: Recipe, Schedule & Optimizer Tuning

## Thesis
In this small-vocab tied-embedding regime, the embedding matrix is unusually load-bearing. The separate TIED_EMBED_LR knob matters more than generic LR sweeps. Small, targeted optimizer stars (not grids) on the tokenizer winner can unlock cheap gains. Additionally, training schedule changes (batch size, logit processing) are zero-artifact-cost recipe improvements drawn from the [NanoGPT speedrun](../references/nanogpt_speedrun_techniques.md).

## Experiments
- **E10**: Tied-embed LR star on tokenizer winner (3-point, ~0.3 H100-hrs)
- **E11**: Matrix/scalar LR star on tokenizer winner (3-point, ~0.3 H100-hrs)
- **E12**: Embedding norm penalty A/B
- **E23**: EMA weight averaging at export (decay sweep: 0.999, 0.9999) — inspired by [Q Labs 10x blog](../references/qlabs_10x_data_efficiency.md). ~20 lines of code. Maintain shadow EMA weights during training, swap in at export time. Targets qgap reduction via smoother weights. Kill if qgap doesn't improve by ≥10% or val_bpb regresses by >0.002. **Unblocked** — depends only on E02, uses in-process roundtrip (not X-05).
- **E28**: Asymmetric logit rescale — different scaling for positive vs negative logits beyond the symmetric tanh softcap (30.0). Speedrun record 41 showed measurable gain. Near-zero artifact cost (1-2 extra scalars). Kill if Δpq ≥ +0.002. Ref: [NanoGPT speedrun 1.9](../references/nanogpt_speedrun_techniques.md#19-asymmetric-logit-rescale). **Unblocked** — depends only on E02.
- **E30**: Batch size schedule — start with smaller batches (more gradient updates early), increase to full size later. Speedrun record 37 (2.358→2.203 min). Zero artifact cost. Modify `grad_accum_steps` at iteration thresholds. Kill if final val_bpb regresses by >0.003. Ref: [NanoGPT speedrun 1.3](../references/nanogpt_speedrun_techniques.md#13-batch-size-schedule). **Unblocked** — depends only on E02.
- **E32**: WSD LR schedule — Warmup-Stable-Decay replacing cosine warmdown. MiniCPM (arXiv:2404.06395) + ICLR 2025 showed WSD outperforms cosine. Three phases: 1% warmup (~140 steps), 75% stable (~10,335 steps), 24% decay (~3,305 steps). Zero artifact cost. Kill if val_bpb regresses >0.003 vs cosine. Ref: [small model landscape](../references/small_model_optimization_landscape.md#1-wsd-learning-rate-schedule-minicpm--tsinghua). **Unblocked** — depends only on E02.
- **E34**: Turbo-Muon — AOL spectral preconditioning for Muon optimizer. arXiv:2502.16982 claims 2.8x orthogonalization speedup. Could reduce our 5 Newton-Schulz steps to 3-4, giving more training steps in 10 min. Zero artifact cost. Kill if quality degrades or step time doesn't improve. Ref: [small model landscape](../references/small_model_optimization_landscape.md#3-turbo-muon-aol-spectral-preconditioning). **Unblocked** — depends only on E02.
- **E35**: Higher β₂ during cooldown — increase Adam β₂ from 0.95 to 0.97-0.99 during the LR decay phase. arXiv:2508.01483 showed this improves final val loss. ~3 lines of code. Can combine with E32 (WSD) or existing cosine warmdown. Kill if val_bpb regresses >0.001. **Unblocked** — depends only on E02.

## Key Metrics
- Δpq (post-roundtrip delta vs baseline)
- qgap (export retention)
- Step time (must not regress significantly)

## Decision Rules
- Promote if Δpq ≤ -0.003 at P1
- Kill if center point remains best (no improvement found)
- Do NOT run the 15-config LR grid — it's explicitly the wrong approach per the PRD

## Status
Not started. Depends on: E09 (tokenizer winner selected).

## Learnings
(Updated as experiments complete)
