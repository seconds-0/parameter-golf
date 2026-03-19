# Track C: Short-Budget Recipe & Optimizer Tuning

## Thesis
In this small-vocab tied-embedding regime, the embedding matrix is unusually load-bearing. The separate TIED_EMBED_LR knob matters more than generic LR sweeps. Small, targeted optimizer stars (not grids) on the tokenizer winner can unlock cheap gains.

## Experiments
- **E10**: Tied-embed LR star on tokenizer winner (3-point, ~0.3 H100-hrs)
- **E11**: Matrix/scalar LR star on tokenizer winner (3-point, ~0.3 H100-hrs)
- **E12**: Embedding norm penalty A/B
- **E23**: EMA weight averaging at export (decay sweep: 0.999, 0.9999) — inspired by [Q Labs 10x blog](../references/qlabs_10x_data_efficiency.md). ~20 lines of code. Maintain shadow EMA weights during training, swap in at export time. Targets qgap reduction via smoother weights. Kill if qgap doesn't improve by ≥10% or val_bpb regresses by >0.002. **Unblocked** — depends only on E02, uses in-process roundtrip (not X-05).

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
