# Track C: Short-Budget Recipe & Optimizer Tuning

## Thesis
In this small-vocab tied-embedding regime, the embedding matrix is unusually load-bearing. The separate TIED_EMBED_LR knob matters more than generic LR sweeps. Small, targeted optimizer stars (not grids) on the tokenizer winner can unlock cheap gains.

## Experiments
- **E10**: Tied-embed LR star on tokenizer winner (3-point, ~0.3 H100-hrs)
- **E11**: Matrix/scalar LR star on tokenizer winner (3-point, ~0.3 H100-hrs)
- **E12**: Embedding norm penalty A/B

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
