# Proxy Calibration Meta-Postmortem

## Why this exists

`CAL-01` exposed a process problem, not just a bad composed stack. The proxy lane was directionally useful, but it let a staged schedule idea (`E30`) promote without ever exercising the later stage that mattered in the real run.

This review captures the stronger rule set we want going forward so the proxy system stays honest.

## What went wrong

- `E30` is a two-stage schedule:
  - phase 1: `131072` tokens
  - phase 2: `524288` tokens
- The short proxy that promoted `E30` ended around `156.5M` tokens and never reached phase 2 of the schedule in a representative way.
- The first real `8xH100` run was therefore the first time the repo tested the full two-stage schedule under realistic throughput.
- `CAL-01` then showed the likely failure region clearly:
  - the stack was competitive early on an equal-tokens basis
  - the curve peaked around the batch transition
  - the run regressed badly after entering the larger-batch regime while WSD was still at max LR

So the process bug was not "we used proxies." The bug was "we allowed a staged idea to promote from a proxy that only validated phase 1."

## New proxy rules

### 1. Staged ideas must be tested across all intended stages before promotion

This applies to:
- batch schedules
- curricula
- delayed regularizers
- cooldown policies
- multi-phase LR schedules

If a proxy does not cross the later stage, it can only be used for an *early-phase ranking* decision, not for full promotion.

### 2. Measure the axis the hypothesis actually changes

For staged schedule ideas, we should compare all of:
- wallclock
- steps
- tokens seen
- validation checkpoints at phase boundaries

Equal-step or equal-wallclock comparisons alone can be misleading when the hypothesis intentionally changes batch size or throughput.

### 3. Separate proxy roles explicitly

- `P1 fast ranking`
  - cheap early signal
  - useful for killing obviously bad ideas
- `phase-aware proxy`
  - must enter later stages
  - required before promoting staged ideas
- `full calibration`
  - validates actual transfer
  - resets the real reference point

### 4. Treat regime mismatches as blockers, not caveats

If a result is promoted under eager fallback but will be deployed under compiled full runs, get at least one compiled datapoint before treating the idea as a trusted full candidate.

### 5. Full runs should be used sooner when the proxy contract is weak

If phase-aware proxies still fail to predict full behavior after this decomposition tranche, then the proxy lane should become:
- good for killing losers
- insufficient by itself for promoting winners

In that world, we should simply do more Runpod full confirmations sooner.

## Current best hypothesis after CAL-01

- `E30` phase 1 is probably genuinely useful
- `E30` phase 2 under the current WSD timing is the likely failure surface
- the proxy lane over-promoted the full schedule because it never validated the later phase
- compile/eager mismatch still matters, but it is not the only plausible explanation anymore

## What changes now

The next tranche should follow this order:

1. same-provider Runpod baseline control
2. full-run-safe watchdog policy
3. phase-aware `E30` proxy that crosses the transition
4. compiled `1xH100` Runpod E30 check
5. full decomposition (`E32`, then `E32 + E28`, then `E30` variants)

If those phase-aware and compiled checks still do not calibrate the proxy lane, we should accept that this repo needs more real `8xH100` runs for promotion decisions than we originally hoped.
