# Track E: Parameter Sharing & Recurrence Moonshots

## Thesis
Share parameters across layer groups and re-spend saved bytes into width or tokenizer slack. High variance, but has first-principles connection to the stored-byte constraint. Only attempt after Tracks A-D plateau.

## Experiments
- **E18**: Grouped layer sharing moonshot

## Key Metrics
- Δpq, qgap, artifact slack (especially freed bytes)
- Step time (sharing may be faster or slower depending on implementation)

## Decision Rules
- Promote only if it beats baseline outright OR frees ≥300kB while staying within 0.003 Δpq
- Kill fast if neither condition met
- This is explicitly a "try once, move on" track

## Status
Not started. Low priority until Tracks A-D exhaust their value.

## Learnings
(Updated as experiments complete)
