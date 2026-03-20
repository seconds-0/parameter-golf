# E23 Post-Mortem: Export-Time EMA

## Decision
Killed for the current implementation and proxy regime.

## Question Tested
Can maintaining EMA shadow weights during training and swapping them in only at export improve export retention on a cheap matched P1 run?

## Matched Baseline / Controls Used
- Control: `phase1_e23_control_p1-20260320-163308-38d0b485`
- Candidates:
  - `phase1_e23_ema0999_p1-20260320-164244-1b5abd14`
  - `phase1_e23_ema09999_p1-20260320-165137-f8faa3a1`
- Same proxy level, same seed, same host class

## Runs Reviewed
- Control postquant: `1.38639890`
- `EMA_DECAY=0.999` postquant: `2.03365982`
- `EMA_DECAY=0.9999` postquant: `5.45891420`

## What Changed in Code / Config
- Config delta was narrow: `EMA_EXPORT=1` plus `EMA_DECAY` in the candidate configs
- Implementation used `ema_export.py` to:
  - initialize fp32 shadow copies of floating-point parameters
  - update them each step
  - replace floating-point parameters in the export state dict at serialization time

## Observed Result
- The branch failed catastrophically on the exported checkpoint
- `0.999` was worse than control by `+0.64726092` postquant
- `0.9999` was worse than control by `+4.07251530` postquant
- `qgap` also worsened materially in both cases

## Why We Believe It
- The failure is too large to be ordinary proxy noise.
- The live prequant path stayed near baseline before the EMA swap, which isolates the failure to the exported snapshot rather than to the ordinary training path.
- The control was matched cleanly and run on the same cheap proxy pattern as the candidates.

## Implementation-Error Check
- The code path did what the experiment claimed: EMA only affected the exported state dict, not the ordinary live training weights.
- That means the conclusion is not “EMA never works”; it is “this export-time EMA design and decay range are wrong for this regime.”
- I do not see evidence of a simple wiring bug such as EMA being silently active during live training or the wrong checkpoint being exported.

## What We Learned
- Long-horizon export-only EMA is not a cheap retention win here.
- The fact that live training stayed healthy but the exported snapshot collapsed is a useful mechanistic result, not just a failed run.
- This branch should not stay in the active composition set.

## Reopen Conditions
- A materially different EMA design is proposed, such as:
  - much shorter-horizon EMA
  - scheduled EMA activation late in training
  - EMA evaluated on a longer training horizon where the shadow weights have time to stabilize
- Reopen only as a new branch, not by reinterpretation of the current result.

## Follow-Up Impact on Queue
`E23` should remain killed in the live queue. It does not block Track C, and it does not justify pausing the current move toward `E28`.
