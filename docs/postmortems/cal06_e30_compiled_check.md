# CAL-06 Post-Mortem: Compiled `1xH100` E30 Check

## Question

Was the original `E30` batch-schedule win mostly an eager-fallback artifact, or does it remain positive on a compiled `1xH100` path?

## Short Answer

`E30` remains decisively positive on the compiled cheap path.

That means the compile-versus-eager mismatch is **not** the primary explanation for `CAL-01`.

## Setup

- Matched same-host `1xH100` bundle on Runpod
- Compiled regime
- Same promoted cheap base for both runs:
  - `E32` WSD
  - `E28(20,30)`
- Same wallclock budget: `300s`
- Same seed: `1337`
- Same hydrated dataset/tokenizer path on the remote host

Runs:
- Control: `phase1_cal06_e30_compiled_control_p1_prime-20260322-162009-969ba44f`
- Candidate: `phase1_cal06_e30_compiled_batch_schedule_p1_prime-20260322-163622-69fbef8d`

## Results

### Control

- prequant `val_bpb`: `1.69517445`
- post-roundtrip `val_bpb`: `1.69715848`
- reloaded post-roundtrip `val_bpb`: `1.69739495`
- `qgap`: `0.00198403`
- train tokens seen: `83,361,792`
- last reported step average: `1890.18 ms`

### Candidate (`E30`)

- prequant `val_bpb`: `1.35620484`
- post-roundtrip `val_bpb`: `1.35948058`
- reloaded post-roundtrip `val_bpb`: `1.35974881`
- `qgap`: `0.00327574`
- train tokens seen: `318,373,888`
- last reported step average: `123.53 ms`

### Matched deltas

- prequant delta: `-0.33896961`
- post-roundtrip delta: `-0.33767790`
- reloaded post-roundtrip delta: `-0.33764614`

## Interpretation

- This is not a borderline result. The compiled `E30` candidate crushed the compiled control on the same host.
- `E30` therefore survives the exact cheap-scale question that mattered most after `CAL-01`: it is **not** only good under eager fallback.
- The result also reinforces the real mechanism behind the original cheap win:
  - `E30` fits dramatically more optimizer steps into the same wallclock
  - it also sees far more tokens in the same budget on `1xH100`
  - that cheap-regime advantage remains real under compilation

## What This Rules Out

- "The original `E30` proxy win was just an eager-fallback artifact."
- "Compiled cheap runs already show `E30` is bad before the full-scale interaction matters."

## What It Does *Not* Prove

- It does not prove that the original full-run `E30` timing is safe on `8xH100`.
- It does not remove the later-stage interaction hypothesis from `CAL-01`.

The likely failure surface is now narrower:
- `E30` phase 1 is genuinely strong
- the original full-run failure is more likely about the later large-batch phase and its interaction with WSD timing

## Decision

- Keep `E30` alive as a serious recipe idea.
- Demote the compile/eager mismatch from "leading suspect" to "important but secondary confound already checked at cheap scale."
- Make the next cheap gating step `CAL-07`, the phase-aware proxy that explicitly crosses the later `E30` boundary.

## Reopen Trigger

Reopen this specific question only if:
- the compiled cheap path changes materially again, or
- later full decomposition shows a direct contradiction that requires rerunning the cheap compiled check.
