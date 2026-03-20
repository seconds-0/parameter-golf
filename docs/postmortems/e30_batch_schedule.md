# E30 Post-Mortem: Batch Schedule

## Decision
Promoted, with a caveat.

The caveat is measurement regime, not result quality: the paired same-host P1 bundle was run under `TORCHDYNAMO_DISABLE=1` because fresh-host `torch.compile` is currently crashing on Vast during warmup. Within that matched eager fallback, the result is decisive enough to promote.

## Question Tested
Does a two-stage batch schedule help this small-model regime by fitting more optimizer updates into the same wallclock?

Tested schedule:
- first `30%` of steps: `TRAIN_BATCH_TOKENS=131072`
- remaining `70%`: `TRAIN_BATCH_TOKENS=524288`

## Matched Baseline / Controls Used
- Control: `phase1_e30_control_p1-20260320-211549-b83629b3`
- Candidate: `phase1_e30_batch_schedule_p1-20260320-212550-76454fcd`

Both runs used:
- same host class: `vast-1xh100`
- same seed: `1337`
- same base recipe: WSD + asymmetric softcap `(20,30)`
- same eager fallback: `TORCHDYNAMO_DISABLE=1`

## Runs Reviewed
- [phase1_e30_control_p1-20260320-211549-b83629b3](/Users/alexanderhuth/Code/oai-param-golf/experiments/results/phase1_e30_control_p1-20260320-211549-b83629b3/metrics.json)
- [phase1_e30_batch_schedule_p1-20260320-212550-76454fcd](/Users/alexanderhuth/Code/oai-param-golf/experiments/results/phase1_e30_batch_schedule_p1-20260320-212550-76454fcd/metrics.json)

## What Changed In Code / Config
- Added `BATCH_SCHEDULE` support tests and the `E30` control/candidate configs.
- Control kept the active WSD + asymmetric `(20,30)` base unchanged.
- Candidate added `BATCH_SCHEDULE="0.3:131072,1.0:524288"`.
- Both configs explicitly used `TORCHDYNAMO_DISABLE=1` because the fresh-host compiled path is currently unstable.

## Observed Result
Control:
- postquant `1.46588267`
- prequant `1.46468347`
- `qgap=0.00119921`
- `393` steps
- `train_tokens_seen=206,045,184`
- `step_avg_ms=765.26`

Candidate:
- postquant `1.41664772`
- prequant `1.41525336`
- `qgap=0.00139435`
- `1194` steps
- `train_tokens_seen=156,499,968`
- `step_avg_ms=251.36`

Primary deltas:
- `Δpq = -0.04923495`
- `Δpre = -0.04943011`
- `Δqgap = +0.00019514`

The candidate processed fewer total tokens but fit roughly `3x` as many optimizer steps into the same wallclock. That step-density gain dominated the small `qgap` regression.

## Why We Believe It
- The comparator was correct: same host class, same seed, same active base recipe, same wallclock, same eager fallback.
- Replay trust remained intact for both runs; raw and roundtrip reload deltas stayed near zero.
- The win is large enough that it is not plausibly noise on a P1 proxy.
- The mechanism matches the logs directly: more steps early, not a mysterious artifact-size trick.

## Implementation-Error Check
- The `BATCH_SCHEDULE` parser and stage selection are covered in infra tests.
- Control and candidate both preserved `supervised_target_fraction=1.0`, so the result is not a masking or ignored-target artifact.
- Batch schedule accounting in the trainer was specifically reviewed during the infra merge cleanup so `train_tokens_seen` reflects the effective batch size used each step.
- The only meaningful caveat is the eager fallback. This experiment did not validate the schedule on the standard compiled path.

## What We Learned
- Early small batches are a strong lever in this regime.
- For this proxy, fitting far more optimizer steps into the same wallclock matters more than maximizing total tokens processed.
- This should become part of the active Track C base for the next cheap side branch.

## Reopen Conditions
- Reconfirm on the standard compiled path once the fresh-host `torch.compile` regression is fixed.
- Reopen only if a later base change materially alters step-time economics or optimizer behavior.

## Follow-Up Impact On Queue
- Promote `E30` into the active Track C base.
- Next Track C branch: `E34a` on top of WSD + asymmetric `(20,30)` + `E30`.
- Keep the compile regression as a separate systems concern; do not quietly mix eager and compiled P1 results in the same comparison bundle.
