# E32 Post-Mortem: WSD Schedule

## Decision
Promoted. WSD is the active base schedule.

## Question Tested
Can replacing the current baseline warmdown schedule with WSD improve quality on the trusted P1 proxy without unacceptable runtime cost?

## Matched Baseline / Controls Used
- Control: `phase1_e32_control_p1-20260320-172313-5aaa1c50`
- Candidate: `phase1_e32_wsd_p1-20260320-173257-a73c389c`
- Same proxy level, same seed, same host class

## Runs Reviewed
- Control prequant/postquant: `1.45614906 / 1.46474630`
- WSD prequant/postquant: `1.43925826 / 1.44033240`
- `Δpre`: `-0.01689080`
- `Δpq`: `-0.02441390`
- `Δqgap`: `-0.00752309`
- Step time: `557.65 ms` → `568.16 ms` (`+1.88%`)

## What Changed in Code / Config
- Config delta was narrow and explicit:
  - `LR_SCHEDULE=wsd`
  - `WSD_WARMUP_FRAC=0.01`
  - `WSD_STABLE_FRAC=0.75`
  - `WSD_DECAY_STYLE=cosine`
- Implementation lived in `train_schedule.py` and fed only the LR schedule path

## Observed Result
- WSD improved both prequant and post-roundtrip quality materially.
- It also collapsed `qgap` from `0.00859724` to `0.00107415`.
- Runtime cost was small enough to treat as noise-level overhead for this kind of win.

## Why We Believe It
- The improvement is large relative to the P1 noise band established earlier.
- The win is broad across prequant, postquant, and qgap, which is much more convincing than a single-metric blip.
- This was a properly matched same-host P1 comparison rather than a loose cross-run comparison.

## Implementation-Error Check
- The config change was exactly the intended hypothesis: swap the schedule, keep the rest of the recipe fixed.
- The implementation path is small and isolated in `train_schedule.py`, which lowers the risk of accidental extra changes.
- I do not see evidence that this branch accidentally bundled another optimizer or export change.

## What We Learned
- WSD is a real recipe improvement for this model family and proxy.
- The benefit is not just export retention; it improves the underlying trained model too.
- WSD is strong enough to serve as the base for follow-on recipe tests.

## Reopen Conditions
- Reopen only if a later composition reveals that WSD interacts badly with a stronger winner, or if an even better schedule candidate replaces it.
- Do not reopen just because a later modifier on top of WSD fails.

## Follow-Up Impact on Queue
This experiment justifies the current ordering: keep WSD as the active base schedule and test follow-on recipe ideas like `E35` and `E28` on top of it.
