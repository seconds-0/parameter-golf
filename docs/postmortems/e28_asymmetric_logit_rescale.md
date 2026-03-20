# E28 Post-Mortem: Asymmetric Logit Rescale

## Decision
Promoted.

## Question Tested
Does replacing the symmetric logit softcap with separate positive and negative caps improve post-roundtrip quality on the promoted WSD base?

## Matched Baseline / Controls Used
- Control: `phase1_e28_control_p1-20260320-192913-2c4b106b`
- Candidates:
  - `phase1_e28_asym3020_p1-20260320-193915-14cc98ae`
  - `phase1_e28_asym3015_p1-20260320-195120-d28513d6`
  - `phase1_e28_asym2030_p1-20260320-202425-1db0062b`

All four runs used the same `proxy_p1_fast`-style budget on the same `1xH100` host with `SEED=1337` and the promoted WSD schedule base.

## Runs Reviewed
- Control: postquant `1.42336352`, prequant `1.42226318`, `qgap=0.00110033`, `511.9 ms/step`
- `(30,20)`: postquant `1.43924815`, prequant `1.43824624`, `qgap=0.00100190`, `572.5 ms/step`
- `(30,15)`: postquant `1.44641556`, prequant `1.44531828`, `qgap=0.00109728`, `613.1 ms/step`
- `(20,30)`: postquant `1.40769697`, prequant `1.40659045`, `qgap=0.00110652`, `512.6 ms/step`

## What Changed In Code / Config
- Added `LOGIT_SOFTCAP_POS` and `LOGIT_SOFTCAP_NEG` support in `train_gpt.py`
- Centralized the softcap transform in `logit_softcap.py`
- Kept replay/export aligned by plumbing the same knobs through `experiments/scripts/export_eval.py`
- Added four P1 configs covering the control plus the three asymmetric points

## Observed Result
The family was not uniformly positive.

- `(30,20)` regressed by `+0.01588463` post-roundtrip
- `(30,15)` regressed by `+0.02305204` post-roundtrip
- `(20,30)` improved by `-0.01566655` post-roundtrip with essentially unchanged step time

So the correct decision is to promote the specific `(20,30)` winner, not “asymmetric caps” in general.

## Why We Believe It
- Same-host matched P1 bundle
- Strong effect size relative to the Track C P1 promotion bar
- The win is not a pure `qgap` trick: prequant also improved
- Runtime stayed flat enough that the result is not just a speed/quality trade

## Implementation-Error Check
- Replay/export trust was already restored before this sweep
- The control and all three candidates completed successfully with full parsed metrics
- The softcap logic was covered by a regression test proving symmetric behavior still matches the old formula and asymmetric behavior really changes the sign-specific saturation
- No unrelated schedule or loader change was bundled into the candidates

## What We Learned
- Positive-heavy asymmetry is harmful in this regime
- Negative-favored asymmetry `(cap_pos=20, cap_neg=30)` is a real cheap recipe win on top of WSD
- This branch changes the active Track C base, so future cheap recipe follow-ups should compose on top of WSD + `(20,30)`

## Reopen Conditions
- Reopen the full softcap sweep only if a later training change materially alters the logit distribution
- Otherwise keep the promoted `(20,30)` point fixed and compose new ideas on top of it

## Follow-Up Impact On Queue
- The active Track C base is now WSD + asymmetric `(20,30)`
- The next cheap Track C branch should be `E30` batch schedule on top of that base
- If switching back to Track B, start with `E24a` rather than resweeping softcaps
