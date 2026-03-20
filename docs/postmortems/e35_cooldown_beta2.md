# E35 Post-Mortem: Higher β₂ During WSD Cooldown

## Decision
Killed on top of the promoted WSD base.

## Question Tested
After `E32` promoted WSD, can raising Adam `β₂` only during the WSD decay phase further improve final quality?

## Matched Baseline / Controls Used
- Control: `phase1_e35_wsd_control_p1-20260320-183746-f3f18702`
- Candidate: `phase1_e35_wsd_beta2_p1-20260320-184735-878125dc`
- Same proxy level, same seed, same host class

## Runs Reviewed
- Control prequant/postquant: `1.44512983 / 1.44615320`
- Candidate prequant/postquant: `1.45651059 / 1.45758252`
- `Δpre`: `+0.01138076`
- `Δpq`: `+0.01142932`
- `qgap`: `0.00102337` → `0.00107193`
- Bytes: `12,756,384` → `12,797,094`
- Step time: `639.8 ms` → `630.1 ms`

## What Changed in Code / Config
- Control stayed on the promoted WSD schedule
- Candidate added only:
  - `ENABLE_COOLDOWN_BETA2=1`
  - `COOLDOWN_BETA2=0.98`
- Implementation in `train_schedule.py` and `train_gpt.py` changed Adam `β₂` only during the WSD decay phase

## Observed Result
- The candidate got slightly faster, but quality got clearly worse.
- Both prequant and post-roundtrip regressed, which is the most important part of the readout.
- The branch did not buy better retention, better bytes, or better artifact slack in exchange.

## Why We Believe It
- This was a properly matched same-host P1 follow-up on the promoted WSD base.
- The loss is much larger than the small runtime gain, and it hits both prequant and postquant.
- The result is exactly the kind of clean negative follow-up that should stop us from overfitting one schedule family.

## Implementation-Error Check
- The code path is narrow: `beta2_for_schedule(...)` returns the altered `β₂` only in the WSD decay phase.
- The control and candidate configs were matched except for the cooldown `β₂` knob.
- I do not see evidence of an accidental extra schedule swap or unrelated optimizer change.

## What We Learned
- The WSD win does not extend automatically to “more optimizer inertia during cooldown.”
- The promoted schedule should stay plain WSD for now.
- The next cheap recipe branch should move to a different mechanism, not a finer re-tuning of this exact schedule family.

## Reopen Conditions
- A materially different cooldown policy is proposed, such as a different `β₂` schedule shape or a different base schedule family.
- Reopen only as a new branch, not as “try one more nearby `β₂` value” on the current WSD base.

## Follow-Up Impact on Queue
This result strengthens the case for moving on to `E28` rather than spending more time on schedule-adjacent tweaks.
