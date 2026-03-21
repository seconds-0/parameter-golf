# E34a Post-Mortem: Polar Express

## Decision
Neutral / no promote.

`PolarExpress-5` was essentially tied on post-roundtrip quality, and `PolarExpress-4` bought a small speed improvement at the cost of a small quality regression. That is useful signal, but not enough to change the active base.

## Question Tested
Can a drop-in replacement for Muon’s Newton-Schulz orthogonalization improve this recipe by either:
- improving post-roundtrip quality at the same step budget, or
- preserving quality while reducing the orthogonalization step count from `5` to `4`

## Matched Baseline / Controls Used
- Control: `phase1_e34a_control_p1-20260320-222847-48fb49f2`
- Candidate A: `phase1_e34a_polarexpress5_p1-20260320-235254-26475b9f`
- Candidate B: `phase1_e34a_polarexpress4_p1-20260321-000155-7b391557`

Both candidates used:
- same host class: `vast-1xh100`
- same seed: `1337`
- same active base recipe: WSD + asymmetric softcap `(20,30)` + `E30`
- same eager fallback measurement regime

## Runs Reviewed
- [phase1_e34a_control_p1-20260320-222847-48fb49f2](/Users/alexanderhuth/Code/oai-param-golf/experiments/results/phase1_e34a_control_p1-20260320-222847-48fb49f2/metrics.json)
- [phase1_e34a_polarexpress5_p1-20260320-235254-26475b9f](/Users/alexanderhuth/Code/oai-param-golf/experiments/results/phase1_e34a_polarexpress5_p1-20260320-235254-26475b9f/metrics.json)
- [phase1_e34a_polarexpress4_p1-20260321-000155-7b391557](/Users/alexanderhuth/Code/oai-param-golf/experiments/results/phase1_e34a_polarexpress4_p1-20260321-000155-7b391557/metrics.json)

## What Changed In Code / Config
- Added `MUON_ORTHO_BACKEND` support so Muon can dispatch between the baseline Newton-Schulz path and Polar Express.
- Added `E34a` control, 5-step, and 4-step configs plus infra validation coverage.
- Kept the promoted WSD + asymmetric `(20,30)` + `E30` base fixed.

## Observed Result
Control:
- postquant `1.40742446`
- prequant `1.40583745`
- `qgap=0.00158701`
- about `235.3 ms/step`

`PolarExpress-5`:
- postquant `1.40728640`
- prequant `1.40575120`
- `qgap=0.00153520`
- about `234.7 ms/step`

`PolarExpress-4`:
- postquant `1.40755582`
- prequant `1.40594214`
- `qgap=0.00161368`
- about `232.2 ms/step`

Primary deltas versus control:
- `PolarExpress-5`: `Δpq = -0.00013806`
- `PolarExpress-4`: `Δpq = +0.00013136`

## Why We Believe It
- The comparator was correct: same host class, same seed, same active base recipe, same measurement regime.
- Replay trust remained intact for all runs; reload deltas stayed near zero.
- The results are too small to support a quality promotion claim.
- The speed/quality trade is internally consistent: fewer steps in the orthogonalizer made the run slightly faster and slightly worse.

## Implementation-Error Check
- The config path is covered by validation tests for the control and both candidate configs.
- The Muon backend dispatch path is covered by an infra test that confirms the Polar Express backend is accepted.
- All three runs preserved `supervised_target_fraction=1.0`, so this was not a masking or data-loader artifact.
- There is no sign of replay/export corruption in this bundle.

## What We Learned
- Polar Express is not a clear recipe win on this stack at P1.
- The optimizer lane is still plausible, though: the 5-step variant stayed essentially tied, and the 4-step variant hints at a small speed/quality trade rather than a hard failure.
- That makes `E34c` a better next optimizer follow-up than either immediate promotion or immediate abandonment of the optimizer lane.

## Reopen Conditions
- Reopen only if a later optimizer improvement suggests the tiny speed/quality trade from `PolarExpress-4` is worth composing or re-measuring on the stronger stack.
- Also reasonable to revisit after a full-run calibration if the optimizer lane looks more valuable at full scale than it does at P1.

## Follow-Up Impact On Queue
- Do not promote `E34a` into the active base.
- Next optimizer follow-up: `E34c`.
- If `E34c` is flat, the next diversification move should be `E24a`.
