# E34c Post-Mortem: NorMuon Adaptive Scaling

## Decision
Neutral / no promote.

## Question Tested
Does NorMuon-style adaptive second-momentum scaling on top of the promoted WSD + asymmetric `(20,30)` + `E30` base improve post-roundtrip quality enough to justify keeping the optimizer lane as the default next branch?

## Matched Baseline / Controls Used
- Control: `phase1_e34c_control_p1-20260321-004949-c4a83875`
- Candidate: `phase1_e34c_normuon_p1-20260321-010303-d12f0bd7`

This was a matched same-host P1 bundle on `vast-1xh100` under the same eager fallback (`TORCHDYNAMO_DISABLE=1`) already used for the active `E30`/`E34a` measurement regime.

## Runs Reviewed
- [Control metrics](../../experiments/results/phase1_e34c_control_p1-20260321-004949-c4a83875/metrics.json)
- [Candidate metrics](../../experiments/results/phase1_e34c_normuon_p1-20260321-010303-d12f0bd7/metrics.json)

## What Changed In Code / Config
- Added `MUON_ADAPTIVE_MODE` and `MUON_ADAPTIVE_BETA2` in `train_gpt.py`
- Added NorMuon-style adaptive scaling after Muon orthogonalization
- Added `phase1_e34c_control_p1.yaml` and `phase1_e34c_normuon_p1.yaml`

One implementation bug surfaced immediately on the first attempt: the adaptive second-momentum buffer was created in bf16, which caused an immediate dtype crash on `lerp_`. That was fixed by keeping the buffer in fp32 and rerunning. Only the rerun counts as evidence.

## Observed Result
- Control: prequant `1.40944965`, postquant `1.41100907`, `qgap=0.00155942`, `242.95 ms/step`
- Candidate: prequant `1.40835540`, postquant `1.40981103`, `qgap=0.00145563`, `247.54 ms/step`

Key deltas versus control:
- `Δpre = -0.00109425`
- `Δpq = -0.00119804`
- `Δqgap = -0.00010379`
- step time `+1.89%`

## Why We Believe It
- The paired control was correct: same host, same proxy, same eager-fallback regime, same base stack
- Replay/export trust was already restored before this experiment
- The rerun completed cleanly and produced internally consistent metrics
- The result direction is coherent: slight prequant improvement, slight postquant improvement, slight qgap improvement, slight runtime cost

## Implementation-Error Check
- The initial crash was real and exposed a bug in the new optimizer path
- That bug was fixed before the counted run by storing the adaptive second moment in fp32
- Local regression coverage now checks both the NorMuon helper and the dtype-mismatch path
- Because the counted run completed normally after the fix, this is not a “tainted by infra” result

## What We Learned
- The optimizer lane is still alive: NorMuon did move the metric in the right direction
- But the effect is too small to promote at P1 under the current rule (`Δpq ≤ -0.003`)
- This is a near-miss, not a dead end. It argues for “possible composition later,” not “default next branch now”
- The cleaner next step is diversification into `E24a`, not stacking another below-threshold Muon tweak immediately

## Reopen Conditions
- Reopen if a later stack amplifies optimizer-quality gains enough that a `~0.001` P1 edge becomes more meaningful
- Reopen if another Muon-side result suggests NorMuon composes unusually well with that change
- Reopen if we want a deeper P2 confirmation bundle specifically for optimizer-lane composition, not as a default next move

## Follow-Up Impact On Queue
- Active base remains WSD + asymmetric `(20,30)` + `E30`
- Next branch should be `E24a`
- `E34b` stays available later if we intentionally revisit Muon internals
