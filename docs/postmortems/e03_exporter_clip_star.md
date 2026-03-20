# E03 Post-Mortem: Exporter Clip-Percentile Star

## Decision
Flat / no win. Keep the default clip percentile and do not spend more time on clip-only sweeps for the current trusted baseline checkpoint.

## Question Tested
Can a pure exporter-only change to `INT8_CLIP_PERCENTILE` improve post-roundtrip quality or bytes on the trusted `E02` checkpoint without changing training?

## Matched Baseline / Controls Used
- Trusted baseline checkpoint: `baseline_repro-20260319-093041-82dade88/final_model.pt`
- Trusted replay sanity gate: `e02_replay_sanity-20260319-190049`
- Bundle reviewed: `e03_clip_star-20260319-190645`

## Runs Reviewed
- `clip_99_995`
- `clip_99_999`
- `clip_99_99984`
- `clip_99_99995`
- `clip_99_99998`

## What Changed in Code / Config
No training path changed. This was a fixed-checkpoint replay through `export_eval.py` with exactly one varying env knob: `INT8_CLIP_PERCENTILE`.

## Observed Result
- Best setting: `99.99998`
- Best postquant: `1.22687860`
- Trusted `E02` postquant: `1.22628807`
- Best `Δpq`: `+0.00059053`
- Default `99.99984` was effectively tied at `1.22687865`
- All tested settings stayed under the byte cap, but none improved quality or bytes enough to promote

## Why We Believe It
- The sweep ran only after replay trust was restored and revalidated on the same baseline checkpoint.
- The default clip value was included in the star, so this was not a “best among arbitrary candidates” situation.
- The whole sweep stayed in a tight but consistently worse band, which is the right shape for a real negative result rather than noisy chaos.

## Implementation-Error Check
- Control choice was correct: fixed checkpoint, single-variable exporter sweep.
- Replay path was trustworthy at this point because `e02_replay_sanity-20260319-190049` matched the in-run `E02` metrics within the accepted tolerance.
- There is no sign of an accidental multi-variable change: the bundle summary shows a clean monotone-ish sweep over one knob.

## What We Learned
- Static clip-percentile tuning is not where the next meaningful win is hiding for the current baseline checkpoint.
- The default clip percentile is already near the local optimum for this exact weight distribution.
- Future Track B effort should move toward training-time distribution shaping rather than more clip-only knob hunting.

## Reopen Conditions
- A winner from `E24`, `E33`, or `E13` changes the weight distribution enough to justify re-sweeping exporter knobs.
- A later composed model meaningfully changes tensor ranges or artifact slack.

## Follow-Up Impact on Queue
This result supports the current queue shape: no more exporter-only clip sweeps until a training-side regularizer or composition changes the distribution.
