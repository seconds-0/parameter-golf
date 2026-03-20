# E04 Post-Mortem: Keep-Float Threshold Star

## Decision
Flat / no win. Do not promote any keep-float threshold change for the current trusted baseline checkpoint.

## Question Tested
Can `INT8_KEEP_FLOAT_MAX_NUMEL` improve post-roundtrip quality or bytes on the trusted `E02` checkpoint after the clip sweep proved flat?

## Matched Baseline / Controls Used
- Trusted baseline checkpoint: `baseline_repro-20260319-093041-82dade88/final_model.pt`
- Fixed clip baseline from `E03`: default `INT8_CLIP_PERCENTILE=99.99984`
- Bundle reviewed: `e04_keep_float_star-20260319-191446`

## Runs Reviewed
- `keep_8192`
- `keep_16384`
- `keep_32768`
- `keep_65536`
- `keep_131072`

## What Changed in Code / Config
No training path changed. This was another fixed-checkpoint replay sweep, this time varying only `INT8_KEEP_FLOAT_MAX_NUMEL`.

## Observed Result
- `8192`, `16384`, `32768`, and `65536` all landed at exactly `1.22687865` postquant with identical bytes
- `131072` improved slightly to `1.22675124`
- But `131072` blew the byte cap badly at `18,054,259` bytes

## Why We Believe It
- This is a very stable negative result: the cap-safe settings were not merely close, they were identical in the bundle summary.
- The only quality improvement came from a setting that clearly violated the artifact constraint, so it is not a hidden promote.

## Implementation-Error Check
- Control choice was correct: same trusted checkpoint, same exporter path, only one knob changed.
- The identical outputs for the cap-safe thresholds are consistent with the threshold being inactive over the same set of tensors in this model, not with parser noise.
- The out-of-cap improvement at `131072` is exactly the kind of pattern expected if more tensors stay float and quality rises for the wrong reason.

## What We Learned
- Keep-float threshold is effectively inert in the currently tested cap-safe range for this checkpoint.
- The only visible quality gain comes from crossing into a regime that the artifact budget cannot support.
- Like `E03`, this pushes Track B toward training-time distribution changes rather than more static exporter stars.

## Reopen Conditions
- A later model or regularizer changes tensor size mix or float/int8 sensitivity enough that the threshold becomes active within the byte cap.
- A future composition creates materially more slack, making a larger keep-float threshold feasible.

## Follow-Up Impact on Queue
This result reinforces the current decision to stop exporter-only knob sweeps on the baseline checkpoint and move to `E24` if Track B becomes active again.
