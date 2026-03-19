# Track D: Lightweight Architecture Rebudgeting

## Thesis
Using existing knobs (NUM_KV_HEADS, MODEL_DIM) to trade stored bytes for quality. Reducing KV heads from 4 to 2 saves bytes that can be re-spent into slightly larger width. No new code needed — config-only changes with high EV.

## Experiments
- **E16**: KV-head rebudget (4→2), config only
- **E17**: KV-head + width rebudget (2 KV heads, larger MODEL_DIM)

## Key Metrics
- Δpq, qgap, artifact slack
- Step time and memory (P3 required if shape changes)
- Parameter count vs quality frontier

## Decision Rules
- E16: promote if Δpq ≤ -0.003 or nearly tied with obvious artifact/runtime savings
- E17: promote if Δpq ≤ -0.005 at P2 and fits runtime
- Both need P3 rehearsal before 8xH100

## Status
Not started. Can start after E02 (baseline reproduction).

## Learnings
(Updated as experiments complete)
