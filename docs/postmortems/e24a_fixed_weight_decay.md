# E24a Post-Mortem: Fixed Weight Decay

## Decision
Killed early after the first nonzero sweep point.

## Question Tested
Would standard fixed L2 weight decay on 2D matrices improve export retention or generalization on top of the promoted `E32` WSD + `E28(20,30)` + `E30` base?

## Matched Baseline / Controls Used
- Control: `phase1_e24a_control_p1-20260321-020005-afea0bac`
- Candidate: `phase1_e24a_wd01_p1-20260321-020914-e9d70fa7`

Both ran on the same `1xH100` Runpod host with the same eager-fallback regime (`TORCHDYNAMO_DISABLE=1`), same seed, same dataset/tokenizer, and the same promoted base recipe.

## Runs Reviewed
- [phase1_e24a_control_p1-20260321-020005-afea0bac](../../experiments/results/phase1_e24a_control_p1-20260321-020005-afea0bac/metrics.json)
- [phase1_e24a_wd01_p1-20260321-020914-e9d70fa7](../../experiments/results/phase1_e24a_wd01_p1-20260321-020914-e9d70fa7/metrics.json)

## What Changed In Code / Config
- Added an env-backed `FIXED_WEIGHT_DECAY` knob in [train_gpt.py](../../train_gpt.py) that multiplies eligible 2D matrix parameters by `1 - lr * wd` after each optimizer step.
- Added matched P1 configs for the control and the first decay point (`wd=0.1`).
- The planned `0.5` and `1.0` points were not run because the first nonzero point already failed by a huge margin.

## Observed Result
- Control landed at prequant `1.41811530`, postquant `1.41965284`, `qgap=0.00153754`.
- `wd=0.1` landed at prequant `1.66554339`, postquant `1.70560576`, `qgap=0.04006236`.
- That is `Δpre=+0.24742809`, `Δpq=+0.28595292`, and `Δqgap=+0.03852482`, far beyond the kill bar.

## Why We Believe It
- The comparison is tightly matched and same-host.
- Replay trust is already established, and both runs emitted clean exact prequant/postquant metrics.
- The result is not a tiny noisy miss; it is a catastrophic regression on both prequant and post-roundtrip quality.

## Implementation-Error Check
- The change is intentionally narrow: a single fixed decay path on the existing promoted stack.
- The result is internally consistent with the mechanism. Artifact bytes shrank drastically, but quality collapsed much more, which is exactly what “over-shrunk weights” would look like here.
- The branch used the measured trainer snapshot with the real `FIXED_WEIGHT_DECAY` codepath, and that codepath has now been restored to the live repo.

## What We Learned
- Standard fixed L2 weight decay is the wrong regularizer shape for this stack at this proxy scale.
- This is not an export-only failure. The prequant path also fell apart, so the branch is conceptually wrong rather than merely overfit to the byte cap.
- If we keep exploring Track B regularization, the next candidates should be more selective (`E24b`, `E33`, or `E13`), not stronger versions of the same fixed decay.

## Reopen Conditions
- Reopen only if we introduce a materially different decay design, or if a much different future stack suggests weight scale is again the main bottleneck.
- Do not reopen by just trying bigger fixed decay values on this same stack.

## Follow-Up Impact On Queue
- `E24a` is closed and dead.
- The next best move is the first real `8xH100` calibration on Runpod using the current promoted stack rather than another proxy regularizer branch by default.
